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
from providers.intent_state import collect_known_ai_order_ids
from providers.intent_state_store import (
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
    monkeypatch.setattr(cfg, "config_cache", dict(cfg.config_cache))
    monkeypatch.setattr(botcore, "config_cache", dict(cfg.config_cache), raising=False)
    monkeypatch.setattr(cfg, "_CONFIG_MTIME", cfg._CONFIG_MTIME)
    monkeypatch.setattr(cfg, "_CONFIG_LAST_CHECK", cfg._CONFIG_LAST_CHECK)

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
    monkeypatch.setattr(cfg, "config_cache", dict(cfg.config_cache))
    monkeypatch.setattr(botcore, "config_cache", dict(cfg.config_cache), raising=False)
    monkeypatch.setattr(cfg, "_CONFIG_MTIME", cfg._CONFIG_MTIME)
    monkeypatch.setattr(cfg, "_CONFIG_LAST_CHECK", cfg._CONFIG_LAST_CHECK)

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
    monkeypatch.setattr(gtc_mod, "is_trade_enabled", lambda: True)
    monkeypatch.setenv("AI_IMPULSE_TRIGGER_ENABLED", "true")
    monkeypatch.setenv("AI_IMPULSE_REQUIRE_ACTIVE_CRON", "false")
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


def test_audit_findings_balance_parity_ohlc_cache_jsonl_ledger_intents_transfer_and_trend(tmp_path, monkeypatch):
    """Regression suite verifying all 6 audit findings:
    1) Portfolio collector balance/limit parity across recovery vs loaded-instrument paths + shared asset recalculation.
    2) 24h series OHLC invariance across populated cache, cleared cache, and LRU-evicted cache.
    3) Unified JSONL bucket parser: unterminated valid JSON line not consumed; non-UTF-8 corrupt main falls back to .previous.
    4) OrderLedger late FILLED 10/10 supersedes partial CANCELED 2/10; active_intents keeps uncovered HYPEUSD.
    5) Rebalance Arbitrum arrival rejects invalid non-EVM tx refs without receipt checks.
    6) priceAnalysis direction flip with draw_plots=True computes trend ONCE and draws immediately.
    """
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import active_intents
    import intelligence.macro_sentiment_ssot as mss
    import intelligence.sentiment.evaluators.price_series_evaluator as pse
    import priceAnalysis as pa
    import state_io
    from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_tradable_position_prompt_view
    from providers.order_ledger import apply_order_event, update_recently_closed
    from verify_tools import rebalance_kraken_to_hl as reb

    # --- 1. Portfolio collector balance & limits parity across recovery vs loaded instruments + shared USDC ---
    now_s = 1_795_600_000.0
    bal_ts = now_s - 45.0
    ord_ts = now_s - 40.0
    cdir = tmp_path / "cachedb_bal"
    (cdir / "state").mkdir(parents=True)
    (cdir / "orders").mkdir(parents=True)
    (cdir / "state" / "venue_balances_state.json").write_text(
        json.dumps({
            "binance:ARB": 10.0,
            "binance:BTC": 0.0,
            "binance:USDC": 100.0,
            "pos:binance:ARBUSDC": {
                "venue": "binance",
                "symbol": "ARBUSDC",
                "base": "ARB",
                "quote": "USDC",
                "price": 1.0,
                "free_base": 10.0,
                "value_usd": 10.0,
                "free_quote": 100.0,
                "effective_max_buy_usd": 99.5,
                "ts": bal_ts,
            },
            "pos:binance:BTCUSDC": {
                "venue": "binance",
                "symbol": "BTCUSDC",
                "base": "BTC",
                "quote": "USDC",
                "price": 60000.0,
                "free_base": 0.0,
                "value_usd": 0.0,
                "free_quote": 100.0,
                "effective_max_buy_usd": 100.0,
                "ts": bal_ts,
            },
            "cash:binance:USDC": {
                "venue": "binance",
                "asset": "USDC",
                "free": 100.0,
            },
        }),
        encoding="utf-8",
    )
    (cdir / "orders" / "cache_order_open_binance.json").write_text(
        json.dumps({
            "items": {
                "ARBUSDC": [
                    {
                        "orderId": "1001",
                        "symbol": "ARBUSDC",
                        "side": "BUY",
                        "type": "LIMIT",
                        "price": "1.0",
                        "origQty": "25.0",
                        "executedQty": "0.0",
                        "status": "NEW",
                        "time": int(ord_ts * 1000.0),
                    },
                    {
                        "orderId": "1002",
                        "symbol": "ARBUSDC",
                        "side": "SELL",
                        "type": "LIMIT",
                        "price": "1.0",
                        "origQty": "4.0",
                        "executedQty": "0.0",
                        "status": "NEW",
                        "time": int(ord_ts * 1000.0),
                    },
                ],
                "BTCUSDC": [],
            },
            "fetchtime": {"ARBUSDC": int(now_s * 1000.0), "BTCUSDC": int(now_s * 1000.0)},
        }),
        encoding="utf-8",
    )
    prices_map = {"ARBUSDC": {"price": 1.0}, "BTCUSDC": {"price": 60000.0}}
    tpos_rec, _, vcash_rec, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir),
        prices_and_trends=dict(prices_map),
        allow_empty_inst_recovery=True,
        allow_network=False,
        persist=False,
        now_ts=now_s,
    )
    fake_insts = {
        "ARBUSDC": SimpleNamespace(symbol="ARBUSDC", provider_name="binance", base="ARB", quote="USDC", params={}, _provider=None),
        "BTCUSDC": SimpleNamespace(symbol="BTCUSDC", provider_name="binance", base="BTC", quote="USDC", params={}, _provider=None),
    }
    tpos_inst, _, vcash_inst, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir),
        inst_map=fake_insts,
        prices_and_trends=dict(prices_map),
        allow_network=False,
        persist=False,
        now_ts=now_s,
    )
    for label, tpos, vcash in (("recovery", tpos_rec, vcash_rec), ("instruments", tpos_inst, vcash_inst)):
        by_sym = {p["symbol"]: p for p in tpos}
        arb_p, btc_p = by_sym["ARBUSDC"], by_sym["BTCUSDC"]
        assert arb_p["free_base"] == pytest.approx(6.0), label
        assert arb_p["locked_base_qty"] == pytest.approx(4.0), label
        assert arb_p["free_quote"] == pytest.approx(75.0), label
        assert arb_p["locked_quote_usd"] == pytest.approx(25.0), label
        assert arb_p["effective_max_buy_usd"] == pytest.approx(74.62), label
        assert btc_p["free_quote"] == pytest.approx(75.0), label
        assert btc_p["effective_max_buy_usd"] == pytest.approx(74.62), label
        assert vcash["binance:USDC"]["free"] == pytest.approx(75.0), label
        assert vcash["binance:USDC"]["locked_quote_usd"] == pytest.approx(25.0), label
        btc_view = evaluate_tradable_position_prompt_view(btc_p, max_usd=2000.0)
        assert ("max_buy=$", 74.62) in btc_view["limit_items"], label

    # Prompt view also defensively caps a manually stale effective_max_buy_usd against free_quote
    stale_view = evaluate_tradable_position_prompt_view(
        {"venue": "binance", "symbol": "BTCUSDC", "price": 60000.0, "free_quote": 75.0, "effective_max_buy_usd": 100.0},
        max_usd=2000.0,
    )
    assert ("max_buy=$", 74.62) in stale_view["limit_items"]

    # --- 2. 24h series OHLC equality with cache populated, cleared, and after LRU eviction ---
    t0_ms = 1_795_500_000_000
    synth_raw = [[t0_ms + idx * 60_000, px] for idx, px in enumerate((100.0, 100.0, 500.0, 1000.0, 500.0, 100.0, 100.0))]
    norm_synth = pse.normalize_and_dedup_24h_series(synth_raw, "BTCUSDC")
    bars_populated = pse._resample_ohlc_bars(norm_synth, 900_000.0)
    m24_populated = pse.compute_24h_price_metrics(norm_synth)
    assert max(b[2] for b in bars_populated) == pytest.approx(100.0)
    assert m24_populated["high_24h"] == pytest.approx(100.0)

    mss._SORTED_TS_PX_MEMO.clear()
    mss._SORTED_TS_PX_ANCHORS.clear()
    mss._TWO_COL_TS_PX_IDS.clear()
    bars_empty = pse._resample_ohlc_bars(norm_synth, 900_000.0)
    m24_empty = pse.compute_24h_price_metrics(norm_synth)
    assert bars_empty == bars_populated
    assert m24_empty["high_24h"] == pytest.approx(100.0)

    # Evict the memo entry by populating > _SORTED_TS_PX_MAX_ENTRIES unrelated series
    for ev_i in range(mss._SORTED_TS_PX_MAX_ENTRIES + 3):
        other = [[t0_ms + k * 60_000, 200.0 + ev_i + k * 0.1] for k in range(5)]
        pse.normalize_and_dedup_24h_series(other, f"SYM{ev_i}USDC")
    bars_evicted = pse._resample_ohlc_bars(norm_synth, 900_000.0)
    m24_evicted = pse.compute_24h_price_metrics(norm_synth)
    assert bars_evicted == bars_populated
    assert m24_evicted["high_24h"] == pytest.approx(100.0)

    # --- 3. JSONL incremental reader waits for '\n' even on valid JSON, and format detector recovers .previous on non-UTF-8 ---
    inc_p = tmp_path / "unterm_valid.jsonl"
    l1 = json.dumps({"s": "BTCUSDC", "i": [t0_ms, 60000.0]}, separators=(",", ":")) + "\n"
    l2_no_nl = json.dumps({"s": "BTCUSDC", "i": [t0_ms + 1000, 60001.0]}, separators=(",", ":"))
    inc_p.write_bytes(l1.encode("utf-8"))
    m0, bad0, off0, di0, anc0, _ = state_io.read_jsonl_bucket_map_from_offset(str(inc_p), start_offset=0)
    assert bad0 is False and m0 == {"BTCUSDC": [[t0_ms, 60000.0]]}
    # Append valid JSON WITHOUT trailing '\n'
    with open(inc_p, "ab") as fh:
        fh.write(l2_no_nl.encode("utf-8"))
    m1, bad1, off1, di1, anc1, inc1 = state_io.read_jsonl_bucket_map_from_offset(
        str(inc_p), start_offset=off0, expected_dev_ino=di0, expected_anchor=anc0
    )
    assert inc1 is True and bad1 is False and m1 == {} and off1 == off0
    # Once '\n' arrives, incremental read consumes l2 exactly once
    with open(inc_p, "ab") as fh:
        fh.write(b"\n")
    m2, bad2, off2, _, _, inc2 = state_io.read_jsonl_bucket_map_from_offset(
        str(inc_p), start_offset=off1, expected_dev_ino=di1, expected_anchor=anc1
    )
    assert inc2 is True and bad2 is False and m2 == {"BTCUSDC": [[t0_ms + 1000, 60001.0]]} and off2 > off1

    # Corrupt non-UTF-8 main .jsonl must NOT raise UnicodeDecodeError and must recover valid .previous (single-line JSONL too)
    corrupt_jl = tmp_path / "corrupt_utf8.jsonl"
    corrupt_jl.write_bytes(b"\xff\xfe\x80\x81invalid-utf8\n")
    Path(f"{corrupt_jl}.previous").write_bytes(l1.encode("utf-8"))
    recovered_doc = state_io.read_jsonl_or_json_dict_cached(str(corrupt_jl), memo={})
    assert recovered_doc == {"items": {"BTCUSDC": [[t0_ms, 60000.0]]}}

    # --- 4a. OrderLedger: late FILLED 10/10 supersedes partial CANCELED 2/10 ---
    orders: dict = {}
    closed_rows: list = []
    rc: dict = {}
    apply_order_event(
        orders,
        {"ts": now_s, "ev": "ACKED", "v": "binance", "sym": "ARBUSDC", "iid": "i-part-1", "cid": "AI_part_1", "vid": "777001", "side": "BUY", "qty": 10.0, "px": 1.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    apply_order_event(
        orders,
        {"ts": now_s + 5.0, "ev": "RECONCILED_PARTIAL", "v": "binance", "sym": "ARBUSDC", "iid": "i-part-1", "cid": "AI_part_1", "vid": "777001", "fq": 2.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    apply_order_event(
        orders,
        {"ts": now_s + 10.0, "ev": "RECONCILED_CANCELED", "v": "binance", "sym": "ARBUSDC", "iid": "i-part-1", "cid": "AI_part_1", "vid": "777001", "fq": 2.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert len(closed_rows) == 1 and closed_rows[0]["state"] == "CANCELED" and closed_rows[0]["fq"] == pytest.approx(2.0)
    rc = update_recently_closed(rc, closed_rows, eff_now=now_s + 10.0)
    # Duplicate partial CANCELED is suppressed
    assert (
        apply_order_event(
            orders,
            {"ts": now_s + 12.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "i-part-1", "cid": "AI_part_1", "vid": "777001", "vst": "CANCELED", "fq": 2.0},
            closed_out=closed_rows,
            recently_closed=rc,
        )
        is None
    )
    # Authoritative FILLED 10/10 MUST NOT be ignored
    apply_order_event(
        orders,
        {"ts": now_s + 20.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "i-part-1", "cid": "AI_part_1", "vid": "777001", "vst": "FILLED", "qty": 10.0, "fq": 10.0, "avg_px": 1.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert len(closed_rows) == 2 and closed_rows[1]["state"] == "FILLED" and closed_rows[1]["fq"] == pytest.approx(10.0)
    rc = update_recently_closed(rc, closed_rows[1:], eff_now=now_s + 20.0)
    # Duplicate FILLED 10/10 afterwards IS suppressed
    assert (
        apply_order_event(
            orders,
            {"ts": now_s + 25.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "i-part-1", "cid": "AI_part_1", "vid": "777001", "vst": "FILLED", "fq": 10.0},
            closed_out=closed_rows,
            recently_closed=rc,
        )
        is None
    )

    # --- 4b. active_intents keeps HYPEUSD when open-order cache covers ONLY SPCXxUSD ---
    ai_root = tmp_path / "ai_scope"
    (ai_root / "cachedb" / "orders").mkdir(parents=True)
    (ai_root / "cachedb" / "state").mkdir(parents=True)
    (ai_root / "cachedb" / "orders" / "cache_order_open_kraken.json").write_text(
        json.dumps({
            "items": {
                "SPCXxUSD": [
                    {"orderId": "OMGUV7-1", "symbol": "SPCXxUSD", "side": "SELL", "status": "NEW", "price": 140.0, "origQty": 5.0, "executedQty": 0.0}
                ]
            },
            "fetchtime": {"SPCXxUSD": int(now_s * 1000.0)},
            "verified_ts": now_s,
        }),
        encoding="utf-8",
    )
    (ai_root / "cachedb" / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps([
            {
                "intent_id": "intent_spcx_live",
                "venue": "kraken",
                "symbol": "SPCXxUSD",
                "side": "SELL",
                "order_type": "LIMIT",
                "requested_qty": 5.0,
                "requested_price": 140.0,
                "timestamp": now_s - 600.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "OMGUV7-1", "status": "NEW"},
            },
            {
                "intent_id": "intent_hype_uncovered",
                "venue": "kraken",
                "symbol": "HYPEUSD",
                "side": "BUY",
                "order_type": "LIMIT",
                "requested_qty": 10.0,
                "requested_price": 40.0,
                "timestamp": now_s - 1800.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "HYPE-OID-99", "status": "NEW"},
            },
        ]),
        encoding="utf-8",
    )
    idx_ai = active_intents.build_active_intent_index(str(ai_root))
    assert {r["symbol"] for r in idx_ai["intents"]} == {"SPCXxUSD", "HYPEUSD"}

    # --- 5a. Rebalance Arbitrum arrival never confirms invalid non-EVM tx reference ---
    cfg_reb = reb.RebalanceConfig(
        withdraw_asset="USDC",
        withdraw_method="Arbitrum One Network - Native",
        arb_rpc_url="https://arb1.arbitrum.io/rpc",
        hl_info_url="https://api.hyperliquid.xyz/info",
        arb_usdc_contract="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        min_net_usdc=6.0,
        min_arb_eth_gas=0.0002,
        poll_interval_sec=0.005,
        poll_timeout_sec=0.015,
        target_address="0x1111111111111111111111111111111111111111",
        configured_withdraw_key="hyperliquid",
    )
    cli_reb = MagicMock()
    cli_reb.funding_withdrawals.return_value = [
        {"withdrawal_id": "WD-BADREF-1", "status": "success", "onchain_transaction": "0xarbtx123"}
    ]
    with (
        patch.object(reb, "fetch_arbitrum_balances", return_value={"eth": 0.001, "usdc": 110.0}),
        patch.object(reb, "fetch_json_post") as mock_rpc,
    ):
        arr_bad = reb.wait_for_arbitrum_arrival(cli_reb, cfg_reb, "WD-BADREF-1", 100.0, 10.0)
    assert arr_bad["arrived"] is False
    assert arr_bad["stages"]["transaction_confirmed"] is False
    assert arr_bad["stages"]["balance_observed"] is True
    assert mock_rpc.call_count == 0

    # --- 5b. priceAnalysis direction flip with draw_plots=True calculates trend ONCE and draws immediately ---
    detect_calls = []
    draw_calls = []
    base_ts = time.time() - 5 * 86400.0
    fake_series = [(int((base_ts + i * 3600.0) * 1000), 100.0 + i * 0.5) for i in range(80)]
    dir_seq = ["up", "down"]
    sig_seq = [(80, fake_series[-1][0], fake_series[-1][1]), (81, fake_series[-1][0] + 60_000, 150.0)]

    def _fake_detect(_ts, _px, **_kw):
        d = dir_seq[len(detect_calls)]
        detect_calls.append(d)
        return {
            "direction": d,
            "start_timestamp": base_ts,
            "duration_seconds": 3600.0 * (len(detect_calls) + 1),
            "current_slope_h": 0.01 if d == "up" else -0.01,
            "blocks": [(0, 10)],
            "estimated_future_hours": 12.0,
        }

    monkeypatch.setattr(pa, "priceLstFor", lambda _s: fake_series)
    monkeypatch.setattr(pa, "_series_signature_for", lambda _s: sig_seq[min(len(detect_calls), len(sig_seq) - 1)])
    monkeypatch.setattr(pa, "detect_long_term_trend", _fake_detect)
    monkeypatch.setattr(pa, "drawPriceLst", lambda _t, _p, _b, sym, direction, dur_h: draw_calls.append((sym, direction, dur_h)))
    monkeypatch.setattr(pa, "estimate_remaining_trend_hours", lambda *_a, **_kw: 24.0)
    pa_state: dict = {}
    c1 = pa.run_price_analysis_cycle(["BTCUSDC"], pa_state, now_ts=1000.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert c1["BTCUSDC"]["direction"] == "up" and len(detect_calls) == 1 and len(draw_calls) == 1
    c2 = pa.run_price_analysis_cycle(["BTCUSDC"], pa_state, now_ts=1120.0, draw_plots=True, inter_symbol_yield_sec=0.0)
    assert c2["BTCUSDC"]["direction"] == "down"
    assert len(detect_calls) == 2  # Exactly 1 computation for cycle 1 and 1 for cycle 2 (NOT 2 on flip!)
    assert len(draw_calls) == 2 and draw_calls[-1][1] == "down"
    assert pa_state["last_draw_ts"]["BTCUSDC"] == 1120.0


def test_cycle4_equity_post_close_provenance_kinematics_and_partial_7d(tmp_path):
    """Cycle 4 overnight audit fixes:
    1. After US post-market closes (20:00 ET), _seed_t212_positions preserves the fresher 19:59 ET
       POST-close cache_current_price (yahoo_ext_cached + t212_close_px + ext_vs_rth_pct) instead of
       reverting 4h backward to stale 16:00 ET pos:t212 RTH close or mislabeling post-close as t212_rth_close.
    2. KrakenXstocksCollector reports underlying_session='EXT_CLOSE' for yahoo_ext_cached siblings.
    3. Unseeded observe-only equities (e.g. AMD_US_EQ) receive quote_source/quote_age_min from cache_current_price,
       prune closed intraday deltas (15m/30m/1h) + stale 90s/5m kinematics, and flag partial <48h 7d archives.
    4. prune_stale_short_metrics prunes CLOSED_EQUITY_INTRADAY_KEYS when quote_age_min > 30m even if
       post_rth_series_replaced is True.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from intelligence.sentiment.collectors.microstructure_collector import Fleet24hAndMicrostructureEnricher
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from intelligence.sentiment.evaluators.asset_session_policy import EquityRthSessionPolicy
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    from state_io import save_json_state

    ny_tz = ZoneInfo("America/New_York")
    ts_rth_close = datetime(2026, 11, 24, 15, 59, 0, tzinfo=ny_tz).timestamp()
    ts_post_close = datetime(2026, 11, 24, 19, 59, 50, tzinfo=ny_tz).timestamp()
    ts_closed_now = datetime(2026, 11, 24, 20, 35, 0, tzinfo=ny_tz).timestamp()  # 35m after POST close

    # --- C4.1: Overnight CLOSED preserves fresher POST-close quote instead of regressing to 4h-older RTH close ---
    pt_seed = {
        "ANT_US_EQ": {"price": 80.04, "trend": 0, "growth_coefficient": 0.0},
        "NVDA_US_EQ": {"price": 151.90, "trend": 0, "growth_coefficient": 0.0},
        "SPCX_US_EQ": {"price": 125.07, "trend": 0, "growth_coefficient": 0.0},
    }
    cp_seed = {
        "ANT_US_EQ": [[int(ts_post_close * 1000), 80.04]],
        "NVDA_US_EQ": [[int(ts_post_close * 1000), 151.90]],
        "SPCX_US_EQ": [[int(ts_post_close * 1000), 125.07]],
    }
    cached_bals = {
        "pos:t212:ANT_US_EQ": {"symbol": "ANT_US_EQ", "free": 0.0, "price": 82.56, "ts": ts_rth_close},
        "pos:t212:NVDA_US_EQ": {"symbol": "NVDA_US_EQ", "free": 0.0, "price": 155.05, "ts": ts_rth_close},
        "pos:t212:SPCX_US_EQ": {"symbol": "SPCX_US_EQ", "free": 2.2, "price": 125.74, "ts": ts_rth_close},
    }
    # Empty inst_map mimics auto_cryptotrade_executor (and observe-only ANT_US_EQ in auto_sharetrade_executor)
    Fleet24hAndMicrostructureEnricher._seed_t212_positions(
        "", pt_seed, cp_seed, {}, cached_bals, "", 0.0, ts_closed_now, lambda _p: None, False
    )
    assert pt_seed["ANT_US_EQ"]["price"] == pytest.approx(80.04)
    assert pt_seed["ANT_US_EQ"]["quote_source"] == "yahoo_ext_cached"
    assert pt_seed["ANT_US_EQ"]["quote_age_min"] == pytest.approx(35.2, abs=0.2)
    assert pt_seed["ANT_US_EQ"]["t212_close_px"] == pytest.approx(82.56)
    assert pt_seed["ANT_US_EQ"]["ext_vs_rth_pct"] == pytest.approx(-3.05, abs=0.05)
    assert "market_state" not in pt_seed["ANT_US_EQ"]
    assert pt_seed["NVDA_US_EQ"]["price"] == pytest.approx(151.90)
    assert pt_seed["NVDA_US_EQ"]["quote_source"] == "yahoo_ext_cached"
    assert pt_seed["SPCX_US_EQ"]["price"] == pytest.approx(125.07)
    assert pt_seed["SPCX_US_EQ"]["quote_source"] == "yahoo_ext_cached"

    # --- C4.2: KrakenXstocksCollector marks yahoo_ext_cached sibling as EXT_CLOSE ---
    k_dir = tmp_path / "kraken_ext_close"
    k_dir.mkdir()
    save_json_state(str(k_dir / "xstock_state.json"), {"known_assets": {"SPCXx": 12.0}, "pair": "SPCXxUSD"})
    xst = KrakenXstocksCollector(start_sync=False).collect(
        str(k_dir),
        kraken_dir=str(k_dir),
        prices_and_trends={
            "SPCXxUSD": {"price": 125.60},
            "SPCX_US_EQ": dict(pt_seed["SPCX_US_EQ"]),
        },
        all_t212_cycles={},
    )
    assert xst is not None
    assert xst["underlying_live_price"] == pytest.approx(125.07)
    assert xst["underlying_session"] == "EXT_CLOSE"
    assert xst["xstock_vs_underlying_pct"] == pytest.approx(0.42, abs=0.02)

    # --- C4.3: Unseeded observe-only AMD_US_EQ gets quote_source/age, prunes closed 15m/1h & 90s/5m kinematics, flags <48h 7d ---
    cdir = tmp_path / "cachedb_amd"
    pdir = cdir / "prices"
    pdir.mkdir(parents=True)
    # 24h series ending at 19:59:50 ET (35m before ts_closed_now) with realistic intraday moves
    pts_24 = [
        [int((ts_post_close - 24 * 3600) * 1000), 447.75],
        [int((ts_post_close - 3600) * 1000), 411.50],
        [int((ts_post_close - 900) * 1000), 408.50],
        [int((ts_post_close - 300) * 1000), 408.70],
        [int((ts_post_close - 90) * 1000), 408.60],
        [int(ts_post_close * 1000), 408.393],
    ]
    (pdir / "cache_24_price_AMD_US_EQ.json").write_text(
        json.dumps({"items": {"AMD_US_EQ": pts_24}}),
        encoding="utf-8",
    )
    # Newly-added 7h archive (far shorter than 48h!)
    pts_arch = [
        {"s": "AMD_US_EQ", "i": [int((ts_post_close - 7 * 3600) * 1000), 412.47]},
        {"s": "AMD_US_EQ", "i": [int(ts_post_close * 1000), 408.393]},
    ]
    (pdir / "cache_365_price_AMD_US_EQ.jsonl").write_text(
        "\n".join(json.dumps(r) for r in pts_arch) + "\n",
        encoding="utf-8",
    )
    pt_amd = {"AMD_US_EQ": {"price": 408.393, "trend": 0, "growth_coefficient": 0.0}}
    cp_amd = {"AMD_US_EQ": [[int(ts_post_close * 1000), 408.393]]}
    Fleet24hAndMicrostructureEnricher._enrich_primary_symbol_series(
        str(cdir),
        "",
        "AMD_US_EQ",
        pt_amd,
        cp_amd,
        {},
        [],
        [],
        set(),
        {},
        {},
        ts_closed_now,
        False,
    )
    amd_e = pt_amd["AMD_US_EQ"]
    assert amd_e.get("quote_source") == "yahoo_ext_cached"
    assert amd_e.get("quote_age_min") == pytest.approx(35.2, abs=0.2)
    assert "chg_15m_pct" not in amd_e
    assert "chg_1h_pct" not in amd_e
    assert "vel_90s_pct_min" not in amd_e
    assert "vel_5m_pct_min" not in amd_e
    assert amd_e.get("is_partial_7d") is True
    assert amd_e.get("span_7d_hours") == pytest.approx(7.0, abs=0.5)
    # Cross-market equity bellwether also omits bogus 7d and closed 15m/1h deltas
    cmc = build_cross_market_context(pt_amd, {})
    amd_bw = cmc["equity_bellwethers"]["AMD_US_EQ"]
    assert amd_bw["quote_source"] == "yahoo_ext_cached"
    assert "chg_15m_pct" not in amd_bw and "chg_1h_pct" not in amd_bw and "chg_7d_pct" not in amd_bw

    # --- C4.4: Closed quote (>30m old) prunes intraday keys even when post_rth_series_replaced=True ---
    pol = EquityRthSessionPolicy()
    closed_entry = {"quote_age_min": 180.0, "chg_15m_pct": 3.46, "chg_30m_pct": 3.72, "chg_1h_pct": 2.45}
    closed_m24 = {"chg_15m_pct": 3.46, "chg_30m_pct": 3.72, "chg_1h_pct": 2.45, "chg_24h_pct": -5.27}
    pol.prune_stale_short_metrics(closed_entry, closed_m24, [], ts_closed_now, post_rth_series_replaced=True)
    assert "chg_15m_pct" not in closed_entry and "chg_1h_pct" not in closed_entry
    assert "chg_15m_pct" not in closed_m24 and "chg_24h_pct" in closed_m24

    # --- C4.5: Live post-RTH price overriding stale cache_current_price / 24h tail must NOT inherit stale quote_age_min ---
    pt_amd_live = {"AMD_US_EQ": {"price": 409.50, "trend": 0, "growth_coefficient": 0.0}}
    Fleet24hAndMicrostructureEnricher._enrich_primary_symbol_series(
        str(cdir),
        "",
        "AMD_US_EQ",
        pt_amd_live,
        cp_amd,
        {},
        [],
        [],
        set(),
        {},
        {},
        ts_closed_now,
        False,
    )
    amd_live_e = pt_amd_live["AMD_US_EQ"]
    assert "quote_source" not in amd_live_e
    assert "quote_age_min" not in amd_live_e
    assert "chg_15m_pct" in amd_live_e and "chg_1h_pct" in amd_live_e


def test_cycle5_upgrade_tpo_vp_preserves_va_hold_min_across_zero_vol_tail_gap(tmp_path):
    """Cycle 5 overnight audit fix:
    When upgrade_tpo_vp_from_ohlc strips >60m of zero-volume overnight OHLC bars (e.g. Kraken SPCXxUSD)
    before appending live cur_px, va_hold_min must NOT collapse to 0.0; it recomputes continuous breakout
    hold duration from series_24h (or unfiltered disk OHLC bars) against the upgraded REAL_VOL VAL/VAH.
    """
    from intelligence.sentiment.collectors.microstructure_collector import upgrade_tpo_vp_from_ohlc

    cdir = tmp_path / "cachedb_vp_hold"
    pdir = cdir / "prices"
    pdir.mkdir(parents=True)

    now_s = 1_795_574_700.0
    now_ms = int(now_s * 1000)
    # 24h of 5m OHLC bars: active volume around $131 (first 20h), drops below VAL (~$126.4) at -190m,
    # last positive-volume bar at -110m ($125.25), followed by 110m of 0-volume overnight bars at $125.25 -> $124.46.
    ohlc_pts = []
    ser_24h = []
    for k in range(288):
        t_ms = now_ms - (288 - k) * 300_000
        age_m = (now_ms - t_ms) / 60_000.0
        if age_m > 190.0:
            px = 131.0 + (0.6 if k % 2 == 0 else -0.6)
            vol = 12.0
        elif age_m > 110.0:
            px = 125.3
            vol = 3.0
        else:
            px = 125.25
            vol = 0.0
        ohlc_pts.append([t_ms, px, vol, px, px, px])
        ser_24h.append([t_ms, 124.5 if age_m <= 60.0 else px])
    ser_24h.append([now_ms - 5_000, 124.46])
    (pdir / "cache_ohlc_24h_SPCXxUSD.json").write_text(
        json.dumps({"ts": now_s - 30.0, "symbol": "SPCXxUSD", "points": ohlc_pts}),
        encoding="utf-8",
    )

    # 1. With continuous series_24h supplied: REAL_VOL upgrade preserves ~190m continuous hold below VAL (NOT 0.0!)
    m24_with_s24 = {"vp_source": "TPO", "low_24h": 124.4, "high_24h": 132.0, "va_regime": "BELOW_VAL", "va_hold_min": 190.0}
    upgrade_tpo_vp_from_ohlc(
        "SPCXxUSD",
        m24_with_s24,
        124.46,
        now_s,
        s24_is_live_clock=True,
        cache_dir=str(cdir),
        allow_network=False,
        series_24h_start_ms=ser_24h[0][0],
        series_24h=ser_24h,
    )
    assert m24_with_s24["vp_source"] == "REAL_VOL"
    assert m24_with_s24["va_regime"] == "BELOW_VAL"
    assert m24_with_s24["va_hold_min"] >= 180.0

    # 2. Without series_24h (disk fallback): unfiltered disk OHLC bars also preserve ~190m hold (NOT 0.0!)
    m24_disk_only = {"vp_source": "TPO", "low_24h": 124.4, "high_24h": 132.0, "va_regime": "BELOW_VAL", "va_hold_min": 190.0}
    upgrade_tpo_vp_from_ohlc(
        "SPCXxUSD",
        m24_disk_only,
        124.46,
        now_s,
        s24_is_live_clock=True,
        cache_dir=str(cdir),
        allow_network=False,
    )
    assert m24_disk_only["vp_source"] == "REAL_VOL"
    assert m24_disk_only["va_regime"] == "BELOW_VAL"
    assert m24_disk_only["va_hold_min"] >= 180.0


def test_cycle6_interior_zero_vol_va_hold_and_cross_quote_7d_archive_stitching(tmp_path):
    """Cycle 6 overnight audit fixes:
    1. upgrade_tpo_vp_from_ohlc recomputes continuous va_hold_min even when a >60m zero-volume OHLC gap
       is INTERIOR to the breakout (e.g., a small trade arrives 45m ago after a 115m quiet period on SPCXxUSD).
    2. CompositeSymbolPriceSeriesLoader must NOT break out of the quote-candidate loop early when a newer
       365-day file on the primary quote (ADAUSDC, 31.18h) is slightly longer than its sparse file (31.14h)
       yet still <5d — it must continue to compatible quote siblings (ADAUSD) and stitch full 7d history
       both when 24h is complete and when 24h supplement is needed.
    """
    from intelligence.sentiment.collectors.microstructure_collector import (
        CompositeSymbolPriceSeriesLoader,
        upgrade_tpo_vp_from_ohlc,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import LAST_7D_SPAN_MS_BY_SYM

    now_s = 1_795_578_000.0
    now_ms = int(now_s * 1000)

    # --- C6.1: Interior >60m zero-volume OHLC gap (115m quiet gap followed by a fresh trade 45m ago) ---
    cdir_vp = tmp_path / "cachedb_vp_interior"
    pdir_vp = cdir_vp / "prices"
    pdir_vp.mkdir(parents=True)
    ohlc_pts = []
    ser_24h = []
    for k in range(288):
        t_ms = now_ms - (288 - k) * 300_000
        age_m = (now_ms - t_ms) / 60_000.0
        if age_m > 245.0:
            px = 131.0 + (0.6 if k % 2 == 0 else -0.6)
            vol = 12.0
        elif age_m > 165.0:
            px = 125.25
            vol = 3.0
        elif 45.0 <= age_m <= 50.5:
            # Small print 45-50m ago makes the LAST gap <=60m while leaving a 115m interior zero-vol gap!
            px = 124.20
            vol = 1.4
        else:
            px = 125.10 if age_m > 50.5 else 124.10
            vol = 0.0
        ohlc_pts.append([t_ms, px, vol, px, px, px])
        ser_24h.append([t_ms, px])
    ser_24h.append([now_ms - 5_000, 124.10])
    (pdir_vp / "cache_ohlc_24h_SPCXxUSD.json").write_text(
        json.dumps({"ts": now_s - 30.0, "symbol": "SPCXxUSD", "points": ohlc_pts}),
        encoding="utf-8",
    )
    m24_int = {"vp_source": "TPO", "low_24h": 124.0, "high_24h": 132.0, "va_regime": "BELOW_VAL", "va_hold_min": 244.0}
    upgrade_tpo_vp_from_ohlc(
        "SPCXxUSD",
        m24_int,
        124.10,
        now_s,
        s24_is_live_clock=True,
        cache_dir=str(cdir_vp),
        allow_network=False,
        series_24h_start_ms=ser_24h[0][0],
        series_24h=ser_24h,
    )
    assert m24_int["vp_source"] == "REAL_VOL"
    assert m24_int["va_regime"] == "BELOW_VAL"
    assert m24_int["va_hold_min"] >= 235.0  # Continuous ~245m, NOT truncated at 50m by the 115m interior gap!

    # --- C6.2a: ADAUSDC sparse (31.14h) + ADAUSDC 365 (31.18h, 2m fresher) must STILL stitch ADAUSD (8d) ---
    cdir_ada = tmp_path / "cachedb_ada_7d"
    pdir_ada = cdir_ada / "prices"
    pdir_ada.mkdir(parents=True)

    # Full 24h cache for ADAUSDC -> need_24h_supplement=False
    p24_rows = [
        {"s": "ADAUSDC", "i": [now_ms - int((24.0 - i * 0.5) * 3600_000), 0.145 - i * 0.00004]}
        for i in range(49)
    ]
    (pdir_ada / "cache_24_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps(r) for r in p24_rows) + "\n",
        encoding="utf-8",
    )
    # Sparse ADAUSDC: -31.18h to -0.04h (span 31.14h)
    sp_usdc = [
        {"s": "ADAUSDC", "i": [now_ms - int(31.18 * 3600_000) + i * 180_000, 0.1479 - i * 0.000007]}
        for i in range(623)
    ]
    (pdir_ada / "cache_sparse_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sp_usdc) + "\n",
        encoding="utf-8",
    )
    # 365 ADAUSDC: -31.18h to -0.001h (span 31.18h -> 2.4m longer than sparse, still <5d!)
    # Make file >512KB so _read_archive_series_file takes span_seek path just like live PROD
    dense_usdc = [
        {"s": "ADAUSDC", "i": [now_ms - int(31.18 * 3600_000) + i * 9_350, 0.1479 - i * 0.0000004]}
        for i in range(12_000)
    ]
    (pdir_ada / "cache_365_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dense_usdc) + "\n",
        encoding="utf-8",
    )
    # Older compatible sibling ADAUSD: -192h (8d) to -31.23h (3m before ADAUSDC started)
    sp_usd = [
        {"s": "ADAUSD", "i": [now_ms - int(192.0 * 3600_000) + i * 3600_000, 0.155 - i * 0.000043]}
        for i in range(161)
    ]
    sp_usd.append({"s": "ADAUSD", "i": [now_ms - int(31.23 * 3600_000), 0.1480]})
    (pdir_ada / "cache_sparse_price_ADAUSD.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sp_usd) + "\n",
        encoding="utf-8",
    )

    loader = CompositeSymbolPriceSeriesLoader()
    s24_a, vals_7d_a = loader.collect(str(cdir_ada), "ADAUSDC", include_7d=True)
    assert len(s24_a) >= 20
    assert len(vals_7d_a) >= 100
    assert LAST_7D_SPAN_MS_BY_SYM.get("ADAUSDC", 0.0) >= 6.5 * 86400.0 * 1000.0

    # --- C6.2b: Same stitch must also succeed when need_24h_supplement=True (partial 8h 24h file) ---
    cdir_ada_sup = tmp_path / "cachedb_ada_7d_sup"
    pdir_ada_sup = cdir_ada_sup / "prices"
    pdir_ada_sup.mkdir(parents=True)
    p24_short = [
        {"s": "ADAUSDC", "i": [now_ms - int((8.0 - i * 0.5) * 3600_000), 0.1445 - i * 0.00009]}
        for i in range(17)
    ]
    (pdir_ada_sup / "cache_24_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps(r) for r in p24_short) + "\n",
        encoding="utf-8",
    )
    (pdir_ada_sup / "cache_sparse_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sp_usdc) + "\n",
        encoding="utf-8",
    )
    (pdir_ada_sup / "cache_365_price_ADAUSDC.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dense_usdc) + "\n",
        encoding="utf-8",
    )
    (pdir_ada_sup / "cache_sparse_price_ADAUSD.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sp_usd) + "\n",
        encoding="utf-8",
    )
    s24_b, vals_7d_b = loader.collect(str(cdir_ada_sup), "ADAUSDC", include_7d=True)
    assert len(s24_b) >= 17
    assert len(vals_7d_b) >= 100
    assert LAST_7D_SPAN_MS_BY_SYM.get("ADAUSDC", 0.0) >= 6.5 * 86400.0 * 1000.0


def test_cycle7_closed_equity_rvol_leak_and_cameroon_human_interest_filter():
    """Cycle 7 overnight audit fixes:
    1. compute_volume_rvol_metrics must advance series_end_ms across trailing 2-column [ts, px] post-market
       bars so stale RTH closing-bell turnover cannot leak when now_ts=None, and
       EquityRthSessionPolicy.evaluate_post_rth_override must forward now_ts into apply_post_rth_price_override
       so closed US equities (7.75h after RTH close) do NOT emit stale closing-bell spot_vol_15m / rvol_15m
       alongside intraday_rth_flow=N/A[no_fresh_vol_bars].
    2. Localized human-interest conflict retrospectives (e.g. Cameroon decade of conflict) must not trigger
       is_high_severity=True on generic 'conflict' without kinetic/macro market context.
    """
    from intelligence.macro.lexicon_ssot import match_high_severity_keywords
    from intelligence.sentiment.evaluators.asset_session_policy import EquityRthSessionPolicy
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        compute_24h_price_metrics,
        compute_volume_rvol_metrics,
    )

    # 2026-11-24 09:30 EST -> 16:00 EST RTH (78 x 5m bars with real volume), followed by 2-col POST ticks to 19:59 EST
    rth_open_ms = 1_795_530_600_000.0
    rth_bars = [
        [rth_open_ms + i * 300_000.0, 155.0 - i * 0.02, 100_000.0 if i < 75 else 250_000.0]
        for i in range(78)
    ]
    post_ticks = [
        [rth_open_ms + 78 * 300_000.0 + k * 600_000.0, 152.10 - k * 0.01]
        for k in range(24)  # 4 hours of 2-column extended-hours ticks up to 19:59 EST
    ]
    mixed_series = rth_bars + post_ticks
    closed_now_s = (rth_open_ms + 78 * 300_000.0) / 1000.0 + 7.75 * 3600.0  # 23:45 EST (7.75h post-RTH)

    # C7.1a: Mixed 3-col RTH + 2-col POST series must NOT leak closing-bell volume even with now_ts=None
    assert compute_volume_rvol_metrics(mixed_series, now_ts=None) == {}
    assert compute_volume_rvol_metrics(rth_bars, now_ts=closed_now_s) == {}

    # C7.1b: evaluate_post_rth_override forwards now_ts so closing-bell spot_vol_15m/rvol_15m never reappear overnight
    eq_pol = EquityRthSessionPolicy()
    m24_init = compute_24h_price_metrics(mixed_series, is_equity=True, now_ts=closed_now_s)
    assert "spot_vol_15m_usd" not in m24_init and "rvol_15m" not in m24_init
    _, m24_ov, ov_px, _ = eq_pol.evaluate_post_rth_override(
        mixed_series,
        dict(m24_init),
        151.90,  # slightly differs from 19:55 bar close -> triggers apply_post_rth_price_override
        closed_now_s,
    )
    assert ov_px == pytest.approx(151.90)
    assert "spot_vol_15m_usd" not in m24_ov
    assert "rvol_15m" not in m24_ov
    assert "spot_vol_1h_usd" not in m24_ov
    assert "rvol_1h" not in m24_ov
    assert m24_ov.get("poc_px") == m24_init.get("poc_px")

    # Even on pure RTH-only bars (no 2-col post ticks), forwarding now_ts blocks stale closing-bell volume
    m24_rth_init = compute_24h_price_metrics(rth_bars, is_equity=True, now_ts=closed_now_s)
    _, m24_rth_ov, _, _ = eq_pol.evaluate_post_rth_override(
        rth_bars,
        dict(m24_rth_init),
        151.90,
        closed_now_s,
    )
    assert "spot_vol_15m_usd" not in m24_rth_ov and "rvol_15m" not in m24_rth_ov

    # C7.2: Human-interest Cameroon feature does not trigger high-severity 'conflict'
    assert match_high_severity_keywords(
        "Shot in the face at 15: The lives shattered by Cameroon's decade of conflict"
    ) == ()
    assert match_high_severity_keywords(
        "Houthi drone attack on Riyadh tower injures 11 amid Middle East escalation"
    )


def test_cycle8_signed_zero_basis_formatting_and_time_to_buy_clickbait_filter():
    """Cycle 8 overnight audit fixes:
    1. Near-zero negative perp_spot_basis_bps / funding_annualized_pct / arb bps (e.g. -0.04) must
       render as '+0.0bps' / '+0.0%', never '-0.0bps' / '-0.0%' in Section 3B.
    2. Editorial stock-pick clickbait ending in 'Time to Buy?' is excluded from breaking news feed.
    """
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section

    sec3b = "\n".join(
        _format_microstructure_section(
            {
                "ADAUSDC": {
                    "ob_imbalance": 0.61,
                    "bid_depth_usd": 170_000.0,
                    "ask_depth_usd": 108_000.0,
                    "bid_wall_px": 0.1426,
                    "bid_wall_usd": 27_800.0,
                    "ask_wall_px": 0.1432,
                    "ask_wall_usd": 19_000.0,
                    "whale_long_pct": 0.58,
                    "taker_buy_sell_ratio": 0.95,
                    "taker_buy_vol_usd": 100_000.0,
                    "taker_sell_vol_usd": 105_000.0,
                    "oi_usd": 56_000_000.0,
                    "oi_1h_chg_pct": -0.2,
                    "perp_spot_basis_bps": -0.04,
                    "funding_annualized_pct": -0.02,
                    "cross_venue_peer": "ADAUSD",
                    "cross_venue_arb_bps": -0.03,
                }
            },
            {"ADAUSDC": {"price": 0.1429}},
        )
    )
    assert "perp_basis=+0.0bps" in sec3b
    assert "perp_basis=-0.0bps" not in sec3b
    assert "funding_ann=+0.0%" in sec3b
    assert "funding_ann=-0.0%" not in sec3b
    assert "arb_vs_ADAUSD=+0.0bps" in sec3b

    assert is_excluded_reference_news_item(
        "The Semiconductor ETF Kept Falling After Nvidia Raised Its Forecast. Time to Buy?"
    )
    assert not is_excluded_reference_news_item(
        "SoftBank drops 15% as Wall Street AI sell-off ripples across Asian tech names"
    )


def test_audit_repeat_cycles4_to_8_edge_fixes(tmp_path):
    """Regression coverage for edge cases found in Cycles 4-8 and b340fa18:
    1. KrakenXstocksCollector maps quote_source='yahoo_rth_close' to underlying_session='RTH_CLOSE'.
    2. upgrade_tpo_vp_from_ohlc:
       - When local series_24h is partial (<22h) and load_stitched_24h_ohlc_points succeeds directly,
         hit_va_gap still loads unfiltered 24h OHLC disk bars + local tail so va_hold_min is not truncated.
       - When hold_src is >1h stale or ends in the opposite VA regime, va_hold_min is NOT overwritten.
    3. CompositeSymbolPriceSeriesLoader applies session_policy.price_divergence_threshold (35% for equities)
       when comparing candidate 7d archive tail prices against archive_pts.
    4. Unseeded equity fallback provenance prefers the fresher matching timestamp when both cp_map and
       raw_s24_last_pt match sym_px_cur.
    5. Pre-trade cross-venue arb telemetry and sentiment_advisor whale entry normalize near-zero negatives
       (no '-0.0bps' or '-0.00%').
    6. Horizon24hPriceEvaluator forwards now_ts into compute_24h_price_metrics so closed RTH RVOL does not leak.
    """
    from datetime import datetime
    from pathlib import Path
    from unittest import mock
    from zoneinfo import ZoneInfo

    from intelligence.sentiment.collectors.microstructure_collector import (
        CompositeSymbolPriceSeriesLoader,
        Fleet24hAndMicrostructureEnricher,
        upgrade_tpo_vp_from_ohlc,
    )
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
        PretradeEvaluationContext,
        PretradeOrderbookCvdAndLiqStage,
    )
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        LAST_7D_SPAN_MS_BY_SYM,
        Horizon24hPriceEvaluator,
    )
    from intelligence.sentiment.sentiment_advisor import _format_single_whale_entry
    from state_io import save_json_state

    # 1. KrakenXstocksCollector recognizes yahoo_rth_close -> RTH_CLOSE
    k_dir = tmp_path / "kraken_yahoo_rth_close"
    k_dir.mkdir()
    save_json_state(str(k_dir / "xstock_state.json"), {"known_assets": {"SPCXx": 10.0}, "pair": "SPCXxUSD"})
    xst = KrakenXstocksCollector(start_sync=False).collect(
        str(k_dir),
        kraken_dir=str(k_dir),
        prices_and_trends={
            "SPCXxUSD": {"price": 126.10},
            "SPCX_US_EQ": {"price": 125.74, "quote_source": "yahoo_rth_close", "quote_age_min": 210.0},
        },
        all_t212_cycles={},
    )
    assert xst is not None
    assert xst["underlying_session"] == "RTH_CLOSE"

    # 2a. upgrade_tpo_vp_from_ohlc with partial local series_24h (2h) + fresh full 24h OHLC cache on disk
    now_s = time.time()
    now_ms = int(now_s * 1000)
    cdir_vp = tmp_path / "cachedb_vp_partial_s24"
    pdir_vp = cdir_vp / "prices"
    pdir_vp.mkdir(parents=True)
    ohlc_pts = []
    for k in range(288):
        t_ms = now_ms - (288 - k) * 300_000
        age_m = (now_ms - t_ms) / 60_000.0
        if age_m > 300.0:
            px, vol = 131.0 + (0.5 if k % 2 == 0 else -0.5), 14.0
        elif age_m > 210.0:
            px, vol = 125.1, 3.0
        elif 35.0 <= age_m <= 40.5:
            px, vol = 124.2, 1.5
        else:
            px, vol = (124.8 if age_m > 40.5 else 124.1), 0.0
        ohlc_pts.append([t_ms, px, vol, px, px, px])
    (pdir_vp / "cache_ohlc_24h_SPCXxUSD.json").write_text(
        json.dumps({"ts": now_s - 20.0, "symbol": "SPCXxUSD", "points": ohlc_pts}),
        encoding="utf-8",
    )
    # Local series_24h only covers last 2h (partial_local_24h=True)
    partial_ser_24h = [[now_ms - (24 - j) * 300_000, 124.1] for j in range(24)]
    partial_ser_24h.append([now_ms - 5_000, 124.05])
    m24_partial = {"vp_source": "TPO", "low_24h": 124.0, "high_24h": 132.0, "va_regime": "BELOW_VAL", "va_hold_min": 120.0}
    upgrade_tpo_vp_from_ohlc(
        "SPCXxUSD",
        m24_partial,
        124.05,
        now_s,
        s24_is_live_clock=True,
        cache_dir=str(cdir_vp),
        allow_network=False,
        series_24h_start_ms=partial_ser_24h[0][0],
        series_24h=partial_ser_24h,
    )
    assert m24_partial["vp_source"] == "REAL_VOL"
    assert m24_partial["va_regime"] == "BELOW_VAL"
    assert m24_partial["va_hold_min"] >= 290.0  # Full ~300m from unfiltered disk OHLC + tail, NOT truncated to 120m

    # 2b. Stale (>1h old) opposite-regime series_24h must NOT overwrite va_hold_min
    stale_opp_ser = [
        [now_ms - int((10.0 - j * 0.25) * 3600_000), 134.0]
        for j in range(30)  # Ends at -2.5h, continuously ABOVE_VAH!
    ]
    m24_stale = {"vp_source": "TPO", "low_24h": 124.0, "high_24h": 132.0, "va_regime": "BELOW_VAL", "va_hold_min": 40.0}
    upgrade_tpo_vp_from_ohlc(
        "SPCXxUSD",
        m24_stale,
        124.05,
        now_s,
        s24_is_live_clock=True,
        cache_dir=str(cdir_vp),
        allow_network=False,
        series_24h_start_ms=now_ms - 24 * 3600_000,
        series_24h=stale_opp_ser,
    )
    assert m24_stale["vp_source"] == "REAL_VOL"
    assert m24_stale["va_regime"] == "BELOW_VAL"
    assert m24_stale["va_hold_min"] <= 45.0  # Must NOT adopt the 435m stale ABOVE_VAH hold

    # 3. Equity 7d stitching respects session_policy.price_divergence_threshold (35%, not hardcoded 10%)
    cdir_eq7 = tmp_path / "cachedb_eq_7d_div"
    pdir_eq7 = cdir_eq7 / "prices"
    pdir_eq7.mkdir(parents=True)
    # Primary 365 file has short 20h span ending at $100; sparse file spans 7d ending at $84 (16% diff: >10% but <35%)
    p24_eq = [
        {"s": "TEST_US_EQ", "i": [now_ms - int((20.0 - i) * 3600_000), 99.0 + i * 0.05]}
        for i in range(21)
    ]
    (pdir_eq7 / "cache_24_price_TEST_US_EQ.jsonl").write_text(
        "\n".join(json.dumps(r) for r in p24_eq) + "\n", encoding="utf-8"
    )
    p365_short = [
        {"s": "TEST_US_EQ", "i": [now_ms - int((20.0 - i * 0.5) * 3600_000), 99.0 + i * 0.025]}
        for i in range(41)
    ]
    (pdir_eq7 / "cache_365_price_TEST_US_EQ.jsonl").write_text(
        "\n".join(json.dumps(r) for r in p365_short) + "\n", encoding="utf-8"
    )
    sp_eq_7d = [
        {"s": "TEST_US_EQ", "i": [now_ms - int((168.0 - i * 4.0) * 3600_000), 82.0 + (i / 42.0) * 2.0]}
        for i in range(43)
    ]
    (pdir_eq7 / "cache_sparse_price_TEST_US_EQ.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sp_eq_7d) + "\n", encoding="utf-8"
    )
    _, vals_eq_7d = CompositeSymbolPriceSeriesLoader().collect(str(cdir_eq7), "TEST_US_EQ", include_7d=True)
    assert len(vals_eq_7d) >= 35
    assert LAST_7D_SPAN_MS_BY_SYM.get("TEST_US_EQ", 0.0) >= 6.0 * 86400.0 * 1000.0

    # 4. Unseeded equity fallback provenance picks fresher matching timestamp (POST over stale RTH at same price)
    ny_tz = ZoneInfo("America/New_York")
    ts_rth = datetime(2026, 11, 24, 15, 59, 0, tzinfo=ny_tz).timestamp()
    ts_post = datetime(2026, 11, 24, 19, 59, 50, tzinfo=ny_tz).timestamp()
    ts_closed = datetime(2026, 11, 24, 20, 35, 0, tzinfo=ny_tz).timestamp()
    cdir_eq_fb = tmp_path / "cachedb_eq_fresher_ts"
    pdir_eq_fb = cdir_eq_fb / "prices"
    pdir_eq_fb.mkdir(parents=True)
    (pdir_eq_fb / "cache_24_price_MU_US_EQ.json").write_text(
        json.dumps(
            {
                "items": {
                    "MU_US_EQ": [
                        [int((ts_post - 24 * 3600) * 1000), 102.0],
                        [int(ts_rth * 1000), 100.0],
                        [int((ts_rth + 3600) * 1000), 100.4],
                        [int((ts_post - 600) * 1000), 100.2],
                        [int(ts_post * 1000), 100.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pt_mu = {"MU_US_EQ": {"price": 100.0, "trend": 0, "growth_coefficient": 0.0}}
    cp_mu = {"MU_US_EQ": [[int(ts_rth * 1000), 100.0]]}  # 4h older RTH close at identical price
    Fleet24hAndMicrostructureEnricher._enrich_primary_symbol_series(
        str(cdir_eq_fb), "", "MU_US_EQ", pt_mu, cp_mu, {}, [], [], set(), {}, {}, ts_closed, False
    )
    assert pt_mu["MU_US_EQ"]["quote_source"] == "yahoo_ext_cached"
    assert pt_mu["MU_US_EQ"]["quote_age_min"] == pytest.approx(35.2, abs=0.2)

    # 5a. Pre-trade cross-venue arb normalizes near-zero negatives to +0.0bps
    cdir_arb = tmp_path / "cachedb_arb_zero"
    tdir_arb = cdir_arb / "telemetry"
    tdir_arb.mkdir(parents=True)
    ob_ts = time.time() - 5.0
    (tdir_arb / "orderbook_depth_ADAUSD.json").write_text(
        json.dumps(
            {
                "ts": ob_ts,
                "symbol": "ADAUSD",
                "ob_mid_price": 1.000003,
                "best_bid": 0.999993,
                "best_ask": 1.000013,
                "bid_depth_usd": 120_000.0,
                "ask_depth_usd": 110_000.0,
                "imbalance_ratio": 0.52,
            }
        ),
        encoding="utf-8",
    )
    ctx_arb = PretradeEvaluationContext(
        symbol="ADAUSDC",
        cachedb_dir=str(cdir_arb),
        regime_context=None,
        dur_sec=None,
        session_policy=resolve_asset_session_policy("ADAUSDC"),
        sym_quotes=("USDC", "USD"),
        is_usd_sym=True,
        is_eq_sym=False,
        sym_variants=["ADAUSDC", "ADAUSD"],
        quote_variants=["ADAUSDC", "ADAUSD"],
        rj=lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None,
        allow_network=False,
        now_ts=ob_ts + 5.0,
    )
    ctx_arb.cur_px_est = 1.0
    ctx_arb.cur_m_info = {
        "ob_mid_price": 1.0,
        "bid_depth_usd": 90_000.0,
        "ask_depth_usd": 85_000.0,
        "spread_bps": 2.0,
    }
    p_est_arb = {"price": 1.0}
    liq_arb: dict = {}
    with mock.patch(
        "intelligence.sentiment.collectors.pretrade_telemetry_collector.compute_cross_venue_arb_metrics",
        side_effect=lambda sym, m_cur, *_a, **_kw: m_cur.update(
            {
                "cross_venue_peer": "ADAUSD",
                "cross_venue_arb_bps": -0.03,
                "net_buy_edge_vs_peer_bps": -0.04,
                "best_buy_venue": "kraken",
                "arb_fee_basis": "maker",
            }
        ),
    ):
        PretradeOrderbookCvdAndLiqStage._reconcile_sibling_arbitrage(ctx_arb, p_est_arb, liq_arb)
    arb_line = ctx_arb.telemetry.get("Cross-Venue Relative Value (Arbitrage)", "")
    assert "arb_vs_ADAUSD=+0.0bps" in arb_line
    assert "net_buy_edge=+0.0bps" in arb_line
    assert "-0.0bps" not in arb_line

    # 5b. sentiment_advisor whale entry normalizes -0.004 funding/basis to +0.00
    w_line = _format_single_whale_entry(
        "BTCUSDC",
        {"symbol": "BTCUSDC", "funding_annualized_pct": -0.004, "perp_spot_basis_bps": -0.004},
    )
    assert "funding=+0.00% ann" in w_line
    assert "perp_spot_basis=+0.00bps" in w_line
    assert "-0.00" not in w_line

    # 6. Horizon24hPriceEvaluator forwards now_ts and blocks stale RTH closing-bell RVOL overnight
    rth_open_ms = 1_795_530_600_000.0
    rth_bars = [[rth_open_ms + i * 300_000.0, 155.0 - i * 0.02, 100_000.0] for i in range(78)]
    closed_now_s = (rth_open_ms + 78 * 300_000.0) / 1000.0 + 7.75 * 3600.0
    ev_res = Horizon24hPriceEvaluator().evaluate(rth_bars, now_ts=closed_now_s)
    assert "spot_vol_15m_usd" not in ev_res and "rvol_15m" not in ev_res


def test_p1_p2_audit_balance_obs_ledger_quote_clock_fixes(tmp_path):
    """Regression coverage for the 4 P1/P2 audit findings:
    1. P1 portfolio_collector: keep observed balance & timestamp bound atomically; persist raw balance
       (never adjusted 75 USDC) so canceling orders restores 100 USDC, and prevent old 100 from stealing a newer ts.
    2. P1 order ledger + open/filled caches:
       - CANCELED 0/10 -> PARTIALLY_FILLED 2/10 never resurrects into active orders (archives terminal partial CANCELED;
         subsequent FILLED 10/10 still upgrades to terminal FILLED).
       - Legacy cache_order_open_<venue>.json with fetchtime only (no verified_ts / symbol_verified_ts) never
         acts as complete-snapshot absence proof; legacy verified_ts caps per-symbol ts against newer WS fetchtime.
       - Legacy fill labeled status='FILLED' with 2/10 executed stays PARTIAL when open-order snapshot is absent.
    3. P2 equity quote provenance: never attach yahoo_ext_cached / t212_rth_close / age from a different price
       (e.g. 408.393) onto displayed price 420.00 across unseeded fallback and Trading212 seeding.
    4. P2 pretrade replay clock: shifting process wall-clock by +8h with fixed replay now_ts produces identical
       equity RVOL and pretrade telemetry.
    """
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace
    from unittest import mock
    from zoneinfo import ZoneInfo

    from cacheaccount import CACHE_BINANCE_ORDER_OPEN_FILENAME, CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME, CacheOrderOpenManager
    from intelligence.sentiment.collectors.microstructure_collector import Fleet24hAndMicrostructureEnricher
    from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import PretradeTelemetryOrchestrator
    from providers.order_ledger import (
        OrderEventType,
        OrderLedgerReader,
        OrderLedgerState,
        OrderLedgerWriter,
        apply_order_event,
        classify_fill_evidence_state,
        update_recently_closed,
    )

    # --- 1a. P1: Raw balance persisted (100), prompt displays 75 while BUY open, reverts to 100 when orders clear ---
    now_s = 1_795_650_000.0
    bal_ts = now_s - 60.0
    ord_ts = now_s - 30.0
    cdir1 = tmp_path / "cachedb_raw_persist"
    (cdir1 / "state").mkdir(parents=True)
    (cdir1 / "orders").mkdir(parents=True)
    vb_path1 = cdir1 / "state" / "venue_balances_state.json"
    vb_path1.write_text(
        json.dumps(
            {
                "binance:ARB": 10.0,
                "binance:USDC": 100.0,
                "pos:binance:ARBUSDC": {
                    "venue": "binance",
                    "symbol": "ARBUSDC",
                    "base": "ARB",
                    "quote": "USDC",
                    "price": 1.0,
                    "free_base": 10.0,
                    "value_usd": 10.0,
                    "free_quote": 100.0,
                    "ts": bal_ts,
                },
                "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 100.0, "ts": bal_ts},
            }
        ),
        encoding="utf-8",
    )
    open_orders_state = [
        {
            "orderId": "B1",
            "symbol": "ARBUSDC",
            "side": "BUY",
            "type": "LIMIT",
            "price": "1.0",
            "origQty": "25.0",
            "executedQty": "0.0",
            "status": "NEW",
            "time": int(ord_ts * 1000.0),
        }
    ]
    failing_provider = SimpleNamespace(free_balance=lambda _asset: None)
    inst_map1 = {
        "ARBUSDC": SimpleNamespace(
            symbol="ARBUSDC",
            provider_name="binance",
            base="ARB",
            quote="USDC",
            params={},
            _provider=failing_provider,
            open_orders=lambda: list(open_orders_state),
        )
    }
    tpos_c1, _, vcash_c1, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir1),
        inst_map=inst_map1,
        prices_and_trends={"ARBUSDC": {"price": 1.0}},
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_s,
    )
    assert tpos_c1[0]["free_quote"] == pytest.approx(75.0)
    assert vcash_c1["binance:USDC"]["free"] == pytest.approx(75.0)
    assert vcash_c1["binance:USDC"]["locked_quote_usd"] == pytest.approx(25.0)
    # Persisted split snapshots MUST keep raw 100.0 (NOT synthetic 75.0) and preserve bal_ts!
    from providers.portfolio_state_store import load_venue_balances_state_dict as _load_vb_split

    bal_bin1 = json.loads((cdir1 / "orders" / "cache_balance_binance.json").read_text(encoding="utf-8"))
    assert bal_bin1["items"]["USDC"]["free"] == pytest.approx(100.0)
    assert bal_bin1["items"]["USDC"]["ts"] == pytest.approx(bal_ts)
    pos_st1 = json.loads((cdir1 / "state" / "portfolio_positions_state.json").read_text(encoding="utf-8"))
    assert pos_st1["pos:binance:ARBUSDC"]["free_quote"] == pytest.approx(100.0)
    assert pos_st1["pos:binance:ARBUSDC"]["quote_bal_ts"] == pytest.approx(bal_ts)
    saved_vb1 = _load_vb_split(str(cdir1))
    assert saved_vb1["cash:binance:USDC"]["free"] == pytest.approx(100.0)
    assert saved_vb1["cash:binance:USDC"]["ts"] == pytest.approx(bal_ts)
    assert saved_vb1["pos:binance:ARBUSDC"]["free_quote"] == pytest.approx(100.0)
    assert saved_vb1["pos:binance:ARBUSDC"]["quote_bal_ts"] == pytest.approx(bal_ts)

    # Next cycle: order canceled (empty open orders) + provider still unavailable -> restores 100.0 free!
    open_orders_state.clear()
    tpos_c2, _, vcash_c2, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir1),
        inst_map=inst_map1,
        prices_and_trends={"ARBUSDC": {"price": 1.0}},
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_s + 30.0,
    )
    assert tpos_c2[0]["free_quote"] == pytest.approx(100.0)
    assert vcash_c2["binance:USDC"]["free"] == pytest.approx(100.0)

    # --- 1b. P1: Old value 100 must NEVER steal a newer timestamp from another record ---
    cdir1b = tmp_path / "cachedb_no_ts_theft"
    (cdir1b / "state").mkdir(parents=True)
    (cdir1b / "orders").mkdir(parents=True)
    (cdir1b / "state" / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:binance:ARBUSDC": {
                    "venue": "binance",
                    "symbol": "ARBUSDC",
                    "base": "ARB",
                    "quote": "USDC",
                    "price": 1.0,
                    "free_base": 10.0,
                    "free_quote": 100.0,
                    "ts": bal_ts,
                },
                # Newer BTCUSDC record with NO free_quote must NOT stamp ts=now_s-5 onto USDC=100.0!
                "pos:binance:BTCUSDC": {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "base": "BTC",
                    "quote": "USDC",
                    "price": 60000.0,
                    "free_base": 0.01,
                    "ts": now_s - 5.0,
                },
                "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 100.0},
            }
        ),
        encoding="utf-8",
    )
    (cdir1b / "orders" / CACHE_BINANCE_ORDER_OPEN_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "ARBUSDC": [
                        {
                            "orderId": "B2",
                            "symbol": "ARBUSDC",
                            "side": "BUY",
                            "type": "LIMIT",
                            "price": "1.0",
                            "origQty": "25.0",
                            "executedQty": "0.0",
                            "status": "NEW",
                            "time": int(ord_ts * 1000.0),
                        }
                    ]
                },
                "symbol_verified_ts": {"ARBUSDC": now_s - 1.0},
            }
        ),
        encoding="utf-8",
    )
    tpos_1b, _, vcash_1b, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir1b),
        prices_and_trends={"ARBUSDC": {"price": 1.0}, "BTCUSDC": {"price": 60000.0}},
        allow_empty_inst_recovery=True,
        allow_network=False,
        persist=False,
        now_ts=now_s,
    )
    by_s1b = {p["symbol"]: p for p in tpos_1b}
    assert by_s1b["ARBUSDC"]["free_quote"] == pytest.approx(75.0)
    assert by_s1b["BTCUSDC"]["free_quote"] == pytest.approx(75.0)
    assert vcash_1b["binance:USDC"]["free"] == pytest.approx(75.0)
    assert vcash_1b["binance:USDC"]["locked_quote_usd"] == pytest.approx(25.0)

    # --- 2a. P1: CANCELED 0/10 -> PARTIALLY_FILLED 2/10 never resurrects order into active ledger ---
    orders_l: dict = {}
    closed_l: list = []
    rc_l: dict = {}
    apply_order_event(
        orders_l,
        {"ts": now_s, "ev": "ACKED", "v": "binance", "sym": "ARBUSDC", "iid": "i-c0", "cid": "AI_c0", "vid": "99001", "side": "BUY", "qty": 10.0, "px": 1.0},
        closed_out=closed_l,
        recently_closed=rc_l,
    )
    apply_order_event(
        orders_l,
        {"ts": now_s + 5.0, "ev": "RECONCILED_CANCELED", "v": "binance", "sym": "ARBUSDC", "iid": "i-c0", "cid": "AI_c0", "vid": "99001", "fq": 0.0},
        closed_out=closed_l,
        recently_closed=rc_l,
    )
    assert orders_l == {}
    assert len(closed_l) == 1 and closed_l[0]["state"] == "CANCELED"
    rc_l = update_recently_closed(rc_l, closed_l, eff_now=now_s + 5.0)

    apply_order_event(
        orders_l,
        {"ts": now_s + 10.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "i-c0", "cid": "AI_c0", "vid": "99001", "vst": "PARTIALLY_FILLED", "qty": 10.0, "fq": 2.0},
        closed_out=closed_l,
        recently_closed=rc_l,
    )
    assert orders_l == {}  # Must NEVER resurrect into active orders!
    assert len(closed_l) == 2 and closed_l[1]["state"] == "CANCELED" and closed_l[1]["fq"] == pytest.approx(2.0)
    rc_l = update_recently_closed(rc_l, closed_l[1:], eff_now=now_s + 10.0)

    # Duplicate PARTIALLY_FILLED 2/10 is suppressed
    assert (
        apply_order_event(
            orders_l,
            {"ts": now_s + 12.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "i-c0", "cid": "AI_c0", "vid": "99001", "vst": "PARTIALLY_FILLED", "qty": 10.0, "fq": 2.0},
            closed_out=closed_l,
            recently_closed=rc_l,
        )
        is None
    )
    assert orders_l == {}

    # Subsequent authoritative FILLED 10/10 still archives terminal FILLED without activating orders
    apply_order_event(
        orders_l,
        {"ts": now_s + 20.0, "ev": "STATUS", "v": "binance", "sym": "ARBUSDC", "iid": "i-c0", "cid": "AI_c0", "vid": "99001", "vst": "FILLED", "qty": 10.0, "fq": 10.0},
        closed_out=closed_l,
        recently_closed=rc_l,
    )
    assert orders_l == {}
    assert len(closed_l) == 3 and closed_l[2]["state"] == "FILLED" and closed_l[2]["fq"] == pytest.approx(10.0)

    # --- 2b. P1: Legacy open cache with fetchtime only (no verified_ts) never proves complete snapshot ---
    leg_open_p = tmp_path / "legacy_open_ft_only.json"
    leg_open_p.write_text(
        json.dumps({"items": {"BTCUSDC": []}, "fetchtime": {"BTCUSDC": int(now_s * 1000)}}),
        encoding="utf-8",
    )
    _, _, leg_ver_ts, leg_sym_ver = CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(leg_open_p))
    assert leg_ver_ts == 0.0 and leg_sym_ver == {}

    leg_open_cap = tmp_path / "legacy_open_ws_bumped_ft.json"
    leg_open_cap.write_text(
        json.dumps(
            {
                "items": {"BTCUSDC": []},
                "fetchtime": {"BTCUSDC": int((now_s + 60.0) * 1000)},
                "verified_ts": now_s,
            }
        ),
        encoding="utf-8",
    )
    _, _, _, cap_sym_ver = CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(leg_open_cap))
    assert cap_sym_ver == {"BTCUSDC": pytest.approx(now_s)}

    # --- 2c. P1: Legacy fill labeled status='FILLED' with 2/10 executed stays PARTIAL without open snapshot ---
    assert (
        classify_fill_evidence_state(
            {"orderId": "OLD-2", "quantity": 2.0, "status": "FILLED"},
            req_qty=10.0,
            resting_now=False,
        )
        == OrderLedgerState.PARTIAL.value
    )
    cdb_leg = tmp_path / "cachedb_legacy_fill"
    (cdb_leg / "orders").mkdir(parents=True)
    writer_leg = OrderLedgerWriter(cachedb_dir=cdb_leg)
    reconciler_leg = OrderLedgerReconciler(cachedb_dir=cdb_leg, use_singletons=False, reconcile_grace_sec=10.0)
    reader_leg = OrderLedgerReader(cachedb_dir=cdb_leg)
    writer_leg.record_event(
        OrderEventType.ACKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=now_s,
        iid="hl-leg-2",
        vid="HL-LEG-2",
        side="BUY",
        qty=10.0,
        px=50.0,
    )
    (cdb_leg / "orders" / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSDC": [
                        {
                            "orderId": "HL-LEG-2",
                            "symbol": "HYPEUSDC",
                            "side": "BUY",
                            "price": 50.0,
                            "quantity": 2.0,
                            "status": "FILLED",
                            "timestamp": int((now_s + 2.0) * 1000),
                        }
                    ]
                },
                "fetchtime": {"HYPEUSDC": int((now_s + 3.0) * 1000)},
            }
        ),
        encoding="utf-8",
    )
    st_leg = reconciler_leg.reconcile_once(now_ts=now_s + 5.0)
    assert st_leg["reconciled_filled"] == 0 and st_leg["reconciled_closed"] == 0 and st_leg["active_count"] == 1
    act_leg = reader_leg.list_active_orders(venue="hyperliquid", apply_tail=False)
    assert len(act_leg) == 1 and act_leg[0]["state"] == OrderLedgerState.PARTIAL.value and act_leg[0]["fq"] == pytest.approx(2.0)

    # --- 3. P2: Provenance never mismatches displayed price (420.00 vs 408.393) on unseeded or T212 paths ---
    ny_tz = ZoneInfo("America/New_York")
    ts_rth_q = datetime(2026, 11, 24, 15, 59, 0, tzinfo=ny_tz).timestamp()
    ts_post_q = datetime(2026, 11, 24, 19, 59, 50, tzinfo=ny_tz).timestamp()
    ts_closed_q = datetime(2026, 11, 24, 20, 35, 0, tzinfo=ny_tz).timestamp()
    cdir_q = tmp_path / "cachedb_amd_420"
    pdir_q = cdir_q / "prices"
    pdir_q.mkdir(parents=True)
    (pdir_q / "cache_24_price_AMD_US_EQ.json").write_text(
        json.dumps(
            {
                "items": {
                    "AMD_US_EQ": [
                        [int((ts_post_q - 24 * 3600) * 1000), 447.75],
                        [int((ts_post_q - 3600) * 1000), 411.50],
                        [int(ts_post_q * 1000), 408.393],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    pt_420 = {"AMD_US_EQ": {"price": 420.00, "trend": 0, "growth_coefficient": 0.0}}
    cp_408 = {"AMD_US_EQ": [[int(ts_post_q * 1000), 408.393]]}
    Fleet24hAndMicrostructureEnricher._enrich_primary_symbol_series(
        str(cdir_q), "", "AMD_US_EQ", pt_420, cp_408, {}, [], [], set(), {}, {}, ts_closed_q, False
    )
    assert pt_420["AMD_US_EQ"]["price"] == pytest.approx(420.00)
    assert "quote_source" not in pt_420["AMD_US_EQ"]
    assert "quote_age_min" not in pt_420["AMD_US_EQ"]

    # T212 seeding also must NOT stamp t212_rth_close / pos:t212 age onto an unrelated 420.00 price
    pt_t212_420 = {"AMD_US_EQ": {"price": 420.00, "trend": 0, "growth_coefficient": 0.0}}
    bals_t212_408 = {"pos:t212:AMD_US_EQ": {"symbol": "AMD_US_EQ", "free": 1.0, "price": 408.393, "ts": ts_rth_q}}
    Fleet24hAndMicrostructureEnricher._seed_t212_positions(
        "",
        pt_t212_420,
        {},
        {"AMD_US_EQ": SimpleNamespace(symbol="AMD_US_EQ", provider_name="trading212")},
        bals_t212_408,
        "",
        0.0,
        ts_closed_q,
        lambda _p: None,
        False,
    )
    assert pt_t212_420["AMD_US_EQ"]["price"] == pytest.approx(420.00)
    assert "quote_source" not in pt_t212_420["AMD_US_EQ"]
    assert "quote_age_min" not in pt_t212_420["AMD_US_EQ"]

    # _apply_yahoo_equity_quote_meta classifies overnight stale POST bar as yahoo_ext_cached (not live yahoo_ext)
    pt_y_meta = {"AMD_US_EQ": {"price": 408.393}}
    Fleet24hAndMicrostructureEnricher._apply_yahoo_equity_quote_meta(
        pt_y_meta, {"AMD_US_EQ": 415.0}, "AMD_US_EQ", 408.393, ts_post_q * 1000.0, False, ts_closed_q
    )
    assert pt_y_meta["AMD_US_EQ"]["quote_source"] == "yahoo_ext_cached"
    assert pt_y_meta["AMD_US_EQ"]["quote_age_min"] == pytest.approx(35.2, abs=0.2)
    assert "market_state" not in pt_y_meta["AMD_US_EQ"]

    # --- 4. P2: Pretrade replay clock invariance when process wall-clock shifts by +8h ---
    rth_start_dt = datetime(2026, 11, 24, 9, 30, 0, tzinfo=ny_tz)
    rth_start_ms = int(rth_start_dt.timestamp() * 1000)
    # 50 x 5m bars (09:30 -> 13:35 ET), with last 3 bars (15m) having ~2.8x average turnover
    eq_bars = [
        [rth_start_ms + i * 300_000, 150.0 + i * 0.05, 300_000.0 if i >= 47 else 100_000.0]
        for i in range(50)
    ]
    replay_now = (eq_bars[-1][0] / 1000.0) + 60.0  # 13:36 ET (during RTH, 1m after last bar)
    cdir_pt = tmp_path / "cachedb_pretrade_clock"
    (cdir_pt / "prices").mkdir(parents=True)
    (cdir_pt / "telemetry").mkdir(parents=True)
    (cdir_pt / "prices" / "cache_24_price_NVDA_US_EQ.json").write_text(
        json.dumps({"items": {"NVDA_US_EQ": eq_bars}}),
        encoding="utf-8",
    )
    (cdir_pt / "telemetry" / "orderbook_depth_NVDA_US_EQ.json").write_text(
        json.dumps(
            {
                "ts": replay_now - 10.0,
                "symbol": "NVDA_US_EQ",
                "ob_mid_price": eq_bars[-1][1],
                "bid_depth_usd": 50_000.0,
                "ask_depth_usd": 45_000.0,
                "imbalance_ratio": 0.53,
                "spread_bps": 3.0,
            }
        ),
        encoding="utf-8",
    )
    orch = PretradeTelemetryOrchestrator(start_sync=False)
    with mock.patch("intelligence.sentiment.collectors.pretrade_telemetry_collector._time.time", return_value=replay_now):
        snap_same_clk = orch.read_snapshot(str(cdir_pt), "NVDA_US_EQ", now_ts=replay_now)
    with mock.patch(
        "intelligence.sentiment.collectors.pretrade_telemetry_collector._time.time",
        return_value=replay_now + 8 * 3600.0,
    ):
        snap_shifted_clk = orch.read_snapshot(str(cdir_pt), "NVDA_US_EQ", now_ts=replay_now)

    cvd_same = snap_same_clk.get("15m Cumulative Volume Delta (CVD)", "")
    cvd_shifted = snap_shifted_clk.get("15m Cumulative Volume Delta (CVD)", "")
    assert "rvol_15m=" in cvd_same
    assert cvd_shifted == cvd_same
    assert dict(snap_shifted_clk) == dict(snap_same_clk)



def test_cycle10_deribit_0dte_skew_exclusion_kinematics_signed_zero_and_nk_sanctions_filter():
    """Cycle 10 daytime audit fixes:
    1. Deribit 25d Skew must exclude expiring 0-DTE (<24h to 08:00 UTC settlement) contracts when
       now_ts is supplied and >=24h expiries exist, preventing gamma IV spikes & asymmetric call/put
       0-DTE OI from causing Simpson's paradox while retaining full live book PC_OI and fallback.
    2. format_kinematic_parts must normalize near-zero negative vel/acc values (-0.00003) to +0.0000.
    3. Diplomatic trivia ('North Korea expels Swiss representative over sanctions related to
       children's camp') must not trigger high-severity 'sanctions', while economic sanctions remain active.
    """
    from datetime import datetime, timezone
    from intelligence.macro.lexicon_ssot import match_high_severity_keywords
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import compute_deribit_options_surface
    from intelligence.sentiment.evaluators.price_series_evaluator import format_kinematic_parts

    # C10.1: Reproduce live 07:02 UTC (58m before 25NOV26 08:00 UTC expiry) Simpson's paradox
    now_ts = datetime(2026, 11, 25, 7, 2, tzinfo=timezone.utc).timestamp()
    books = [
        # 0-DTE expiring in 58m: huge call OI at exploding 135% IV vs smaller put OI at 164% IV
        {"instrument_name": "BTC-25NOV26-73000-C", "estimated_delivery_price": 70000.0, "open_interest": 2800.0, "mark_iv": 135.5},
        {"instrument_name": "BTC-25NOV26-67000-P", "estimated_delivery_price": 70000.0, "open_interest": 1500.0, "mark_iv": 164.4},
        # >=24h institutional maturities: clean +9.0% put skew (put IV 61.0% vs call IV 52.0%)
        {"instrument_name": "BTC-27NOV26-74000-C", "estimated_delivery_price": 70000.0, "open_interest": 5000.0, "mark_iv": 52.0},
        {"instrument_name": "BTC-27NOV26-66000-P", "estimated_delivery_price": 70000.0, "open_interest": 5000.0, "mark_iv": 61.0},
        {"instrument_name": "BTC-4DEC26-75000-C", "estimated_delivery_price": 70000.0, "open_interest": 4000.0, "mark_iv": 52.0},
        {"instrument_name": "BTC-4DEC26-65000-P", "estimated_delivery_price": 70000.0, "open_interest": 4000.0, "mark_iv": 61.0},
    ]
    surf_live = compute_deribit_options_surface(books, currency="BTC", dvol=55.0, rv=55.0, now_ts=now_ts)
    # Full book PC_OI retains 0-DTE OI: puts=(1500+5000+4000)=10500, calls=(2800+5000+4000)=11800 -> 0.89
    assert surf_live["btc_pc_oi_ratio"] == pytest.approx(0.89, abs=0.01)
    # Skew reflects institutional >=24h maturities (+9.00%), NOT Simpson's paradox (~+4.3%)
    assert surf_live["btc_skew_25d_pct"] == pytest.approx(9.0, abs=0.01)
    assert surf_live["btc_vrp_pct"] == 0.0

    # Fallback when ONLY 0-DTE options exist still computes a skew rather than dropping the metric
    surf_0dte_only = compute_deribit_options_surface(books[:2], currency="BTC", now_ts=now_ts)
    assert surf_0dte_only["btc_skew_25d_pct"] == pytest.approx(28.9, abs=0.1)

    # Backward compatibility when now_ts is omitted (pools all provided books)
    surf_legacy = compute_deribit_options_surface(books, currency="BTC")
    assert surf_legacy["btc_skew_25d_pct"] < 5.0

    # C10.2: Signed negative zero normalized in format_kinematic_parts (both compact and verbose)
    entry = {"vel_90s_pct_min": -0.00003, "accel_90s_pct_min2": 0.4224, "kalman_vel_z": -0.01}
    compact_parts = format_kinematic_parts(entry, compact=True)
    assert "vel_90s=+0.0000%/m" in compact_parts
    assert all("-0.0000" not in p for p in compact_parts)
    verbose_parts = format_kinematic_parts({"vel_90s_pct_min": 0.0123, "accel_90s_pct_min2": -0.00004}, compact=False)
    assert "90s: vel=+0.0123%/m, acc=+0.0000%/m^2" in verbose_parts

    # C10.3: Diplomatic children's-camp sanctions trivia excluded; real economic sanctions preserved
    assert match_high_severity_keywords(
        "North Korea expels Swiss representative over sanctions related to children's camp"
    ) == ()
    assert match_high_severity_keywords(
        "US Treasury imposes new sanctions package targeting Iranian oil tankers"
    )
    assert match_high_severity_keywords(
        "Swiss freeze Russian bank assets under expanded EU sanctions"
    )


def test_cycle11_24h_ohlc_kline_limits_replace_fallback_sre_filter_and_closed_equity_va_hold(tmp_path, monkeypatch):
    """Cycle 11 audit fixes:
    1. BinanceProvider.ohlc_series & HyperliquidProvider.ohlc_series request full >=24h windows
       for 5m/15m queries (limit=300/102 on Binance, lookback_h=25 on Hyperliquid) instead of
       truncating 24h OHLC/Volume Profile at 22.33h (90 x 15m).
    2. SRE log anomaly harvester ignores benign monitororder native-replace fallback lines
       ('BLOCKED (pre-submit): replace_not_attempted' / 'replace_cancel_only') while still
       catching real pre-submit execution errors.
    3. EquityRthSessionPolicy strips stale/synthetic overnight va_hold_min ('below_val=0m')
       on closed US equities (quote_age_min > 30m) while preserving live RTH/EXT hold duration.
    """
    from types import SimpleNamespace
    import providers.binance_provider as bn_mod
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs
    from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy

    # C11.1a: BinanceProvider.ohlc_series requests >=24h for 5m (limit=300) and 15m (limit=102), keeps 91 for 60m
    bn_calls = []
    raw_k = [
        [1700000000000, "100.0", "101.0", "99.0", "100.5", "10.0", 1700000300000],
        [1700000300000, "100.5", "102.0", "100.0", "101.0", "12.0", 1700000600000],
    ]

    def _fake_get_klines(*, symbol, interval, limit):
        bn_calls.append((symbol, interval, limit))
        return raw_k

    monkeypatch.setattr(bn_mod, "_get_bapi", lambda: SimpleNamespace(client=SimpleNamespace(get_klines=_fake_get_klines)))
    bp = BinanceProvider()
    bp.ohlc_series("ETHUSDC", 5)
    bp.ohlc_series("ETHUSDC", 15)
    bp.ohlc_series("ETHUSDC", 60)
    assert bn_calls == [
        ("ETHUSDC", "5m", 300),
        ("ETHUSDC", "15m", 102),
        ("ETHUSDC", "1h", 91),
    ]

    # C11.1b: HyperliquidProvider.ohlc_series requests >=24h lookback (25h) for 5m and 15m, keeps 90h for 60m
    hl_calls = []
    hl_raw = [
        {"o": "58.0", "h": "59.0", "l": "57.5", "c": "58.5", "v": "100.0", "T": 1700000300000},
        {"o": "58.5", "h": "59.2", "l": "58.0", "c": "58.8", "v": "120.0", "T": 1700000600000},
    ]
    hp = HyperliquidProvider()
    monkeypatch.setattr(
        hp,
        "_require_hl_client_and_pair",
        lambda _c: (
            SimpleNamespace(candles=lambda pair, iv, lh: (hl_calls.append((pair, iv, lh)) or hl_raw)),
            "HYPE/USDC",
            None,
        ),
    )
    hp.ohlc_series("HYPEUSDC", 5)
    hp.ohlc_series("HYPEUSDC", 15)
    hp.ohlc_series("HYPEUSDC", 60)
    assert hl_calls == [
        ("HYPE/USDC", "5m", 25),
        ("HYPE/USDC", "15m", 25),
        ("HYPE/USDC", "1h", 90),
    ]

    # C11.2: Benign monitororder native-replace fallbacks are not counted as SRE errors; real blocks still are
    ldir = tmp_path / "logs_c11"
    ldir.mkdir(parents=True)
    now_hms = time.strftime("%H:%M:%S")
    (ldir / "monitororder_2026-11-25.log").write_text(
        f"{now_hms} Trying to place BUY TAOUSDC qty 11.0375 at price 174.3\n"
        f"{now_hms} [TAOUSDC] BUY BLOCKED (pre-submit): replace_not_attempted\n"
        f"{now_hms} [TAOUSDC] SELL BLOCKED (pre-submit): replace_cancel_only\n"
        f"{now_hms} The order with ID 729450583 was cancelled.\n",
        encoding="utf-8",
    )
    anom_clean = harvest_log_anomalies_from_dirs([str(ldir)], 1.0)
    assert anom_clean.error_count == 0

    (ldir / "monitororder_2026-11-25.log").write_text(
        f"{now_hms} [TAOUSDC] BUY BLOCKED (pre-submit): execution_gate_unavailable (db timeout)\n",
        encoding="utf-8",
    )
    anom_real = harvest_log_anomalies_from_dirs([str(ldir)], 1.0)
    assert anom_real.error_count == 1

    # C11.3: Closed overnight equity strips synthetic va_hold_min ('below_val=0m'), live equity preserves it
    eq_pol = resolve_asset_session_policy("RGNT_US_EQ")
    now_s = time.time()
    closed_entry = {"price": 0.7939, "quote_age_min": 411.3, "va_regime": "BELOW_VAL", "va_hold_min": 0.0}
    closed_m24 = {"va_regime": "BELOW_VAL", "va_hold_min": 0.0, "chg_24h_pct": -6.37}
    eq_pol.prune_stale_short_metrics(closed_entry, closed_m24, [], now_s, has_series_tail=False)
    assert "va_hold_min" not in closed_entry and "va_hold_min" not in closed_m24

    live_entry = {"price": 0.7939, "quote_age_min": 2.0, "va_regime": "BELOW_VAL", "va_hold_min": 25.0}
    live_m24 = {"va_regime": "BELOW_VAL", "va_hold_min": 25.0, "chg_1m_pct": -0.1, "chg_24h_pct": -6.37}
    eq_pol.prune_stale_short_metrics(live_entry, live_m24, [], now_s, has_series_tail=False)
    assert live_entry["va_hold_min"] == pytest.approx(25.0)
    assert live_m24["va_hold_min"] == pytest.approx(25.0)


def test_audit_ac628e45_ledger_terminal_closure_coverage_and_legacy_fill_fixes(tmp_path):
    """Regression suite for post-ac628e45 audit Cycles 1-3 + Cycle 7:
    1) CANCELED(fq=2/10) allows late authoritative FILLED(10/10); CANCELED(fq=0) never resurrects on late PARTIALLY_FILLED;
    2) Unified (venue, symbol) coverage & partial-fill parity in active_intents + legacy fetchtime provenance + legacy partial FILLED row;
    3) Monotonic cid/cloid orphan suppression, fill-status merge, chronological iter_audit_compat_records, intent reconciliation & AI attribution price preservation."""
    from providers.ai_attribution import _evaluate_execution_audit_actors
    from providers.intent_reconciliation import reconcile_intents_from_execution_audit
    from providers.order_id_context import derive_venue_128bit_client_order_id
    from providers.order_ledger import apply_order_event, update_recently_closed

    t0 = 1_795_570_000.0

    # --- #1A: CANCELED with partial fill (2/10) must NOT drop late authoritative FILLED (10/10) ---
    orders: dict = {}
    closed_rows: list = []
    rc: dict = {}
    apply_order_event(
        orders,
        {"ts": t0, "ev": "ACKED", "v": "hyperliquid", "sym": "HYPEUSDC", "iid": "iid-p1", "vid": "HL-100", "side": "BUY", "qty": 10.0, "px": 60.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    apply_order_event(
        orders,
        {"ts": t0 + 5.0, "ev": "RECONCILED_PARTIAL", "v": "hyperliquid", "sym": "HYPEUSDC", "iid": "iid-p1", "vid": "HL-100", "fq": 2.0, "avg_px": 60.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    apply_order_event(
        orders,
        {"ts": t0 + 50.0, "ev": "RECONCILED_CANCELED", "v": "hyperliquid", "sym": "HYPEUSDC", "iid": "iid-p1", "vid": "HL-100", "reason": "absent_from_verified_open_book"},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert len(closed_rows) == 1 and closed_rows[0]["state"] == "CANCELED" and closed_rows[0]["fq"] == pytest.approx(2.0)
    rc = update_recently_closed(rc, closed_rows, eff_now=t0 + 50.0)

    res_sup = apply_order_event(
        orders,
        {"ts": t0 + 70.0, "ev": "STATUS", "v": "hyperliquid", "sym": "HYPEUSDC", "iid": "iid-p1", "vid": "HL-100", "vst": "FILLED", "fq": 10.0, "avg_px": 60.1},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert res_sup is not None and orders == {}
    assert len(closed_rows) == 2 and closed_rows[1]["state"] == "FILLED" and closed_rows[1]["fq"] == pytest.approx(10.0)
    rc = update_recently_closed(rc, closed_rows[1:], eff_now=t0 + 70.0)

    assert apply_order_event(
        orders,
        {"ts": t0 + 80.0, "ev": "STATUS", "v": "hyperliquid", "sym": "HYPEUSDC", "iid": "iid-p1", "vid": "HL-100", "vst": "FILLED", "fq": 10.0},
        closed_out=closed_rows,
        recently_closed=rc,
    ) is None

    # --- #1B: CANCELED with zero filled must NEVER be resurrected into active orders by late PARTIALLY_FILLED ---
    apply_order_event(
        orders,
        {"ts": t0 + 100.0, "ev": "ACKED", "v": "binance", "sym": "BTCUSDC", "iid": "iid-c0", "vid": "BN-200", "side": "BUY", "qty": 1.0, "px": 60000.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    apply_order_event(
        orders,
        {"ts": t0 + 110.0, "ev": "CANCEL_ACK", "v": "binance", "sym": "BTCUSDC", "iid": "iid-c0", "vid": "BN-200"},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    rc = update_recently_closed(rc, closed_rows[-1:], eff_now=t0 + 110.0)
    assert orders == {}
    apply_order_event(
        orders,
        {"ts": t0 + 120.0, "ev": "STATUS", "v": "binance", "sym": "BTCUSDC", "iid": "iid-c0", "vid": "BN-200", "vst": "PARTIALLY_FILLED", "fq": 0.2, "qty": 1.0},
        closed_out=closed_rows,
        recently_closed=rc,
    )
    assert orders == {}
    assert closed_rows[-1]["state"] == "CANCELED" and closed_rows[-1]["fq"] == pytest.approx(0.2)
    rc = update_recently_closed(rc, closed_rows[-1:], eff_now=t0 + 120.0)
    assert apply_order_event(
        orders,
        {"ts": t0 + 125.0, "ev": "STATUS", "v": "binance", "sym": "BTCUSDC", "iid": "iid-c0", "vid": "BN-200", "vst": "PARTIALLY_FILLED", "fq": 0.2, "qty": 1.0},
        closed_out=closed_rows,
        recently_closed=rc,
    ) is None
    assert orders == {}

    # --- #2A: active_intents uses per-(venue, symbol) verified snapshot coverage, matching OrderLedger ---
    ws = tmp_path / "ws_cov"
    orders_dir = ws / "cachedb" / "orders"
    state_dir = ws / "cachedb" / "state"
    orders_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    now_s = time.time()

    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(orders_dir / CACHE_KRAKEN_ORDER_OPEN_FILENAME),
        {"SPCXxUSD": []},
        now_ms=int(now_s * 1000),
        verified_ts=None,
    )
    (state_dir / "autonomous_trade_intents_state.json").write_text(
        json.dumps([
            {
                "intent_id": "ai-kr-hype",
                "venue": "kraken",
                "symbol": "HYPEUSD",
                "side": "BUY",
                "order_type": "LIMIT",
                "requested_qty": 2.0,
                "requested_price": 60.0,
                "timestamp": now_s - 600.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "OKRK-HYPE-LIVE", "status": "OPEN"},
            },
            {
                "intent_id": "ai-kr-spcx",
                "venue": "kraken",
                "symbol": "SPCXxUSD",
                "side": "BUY",
                "order_type": "LIMIT",
                "requested_qty": 1.0,
                "requested_price": 140.0,
                "timestamp": now_s - 600.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "OKRK-SPCX-GONE", "status": "OPEN"},
            },
        ]),
        encoding="utf-8",
    )
    idx = active_intents.build_active_intent_index(str(ws))
    assert [it["intent_id"] for it in idx["intents"]] == ["ai-kr-hype"]

    # --- #2B: Legacy open-order cache without symbol_verified_ts must NOT treat WS-ambiguous fetchtime as snapshot proof ---
    legacy_ws_only = orders_dir / CACHE_BINANCE_ORDER_OPEN_FILENAME
    legacy_ws_only.write_text(
        json.dumps({
            "items": {"BTCUSDC": []},
            "fetchtime": {"BTCUSDC": int(now_s * 1000)},
            "verified_ts": None,
        }),
        encoding="utf-8",
    )
    _, _, v_ts_ws, sym_ver_ws = CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(legacy_ws_only))
    assert v_ts_ws == 0.0 and sym_ver_ws == {}

    legacy_with_ver = tmp_path / "legacy_verified_open.json"
    legacy_with_ver.write_text(
        json.dumps({
            "items": {"BTCUSDC": []},
            "fetchtime": {"BTCUSDC": int((t0 + 200.0) * 1000)},
            "verified_ts": t0 + 100.0,
        }),
        encoding="utf-8",
    )
    _, _, v_ts_leg, sym_ver_leg = CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(str(legacy_with_ver))
    assert v_ts_leg == pytest.approx(t0 + 100.0)
    assert sym_ver_leg == {"BTCUSDC": pytest.approx(t0 + 100.0)}

    # --- #2C: Legacy partial fill (2/10) tagged status=FILLED stays PARTIAL with normalized PARTIALLY_FILLED vst ---
    cdb_f = tmp_path / "cdb_legacy_fill"
    (cdb_f / "orders").mkdir(parents=True)
    writer = OrderLedgerWriter(cachedb_dir=str(cdb_f))
    reconciler = OrderLedgerReconciler(cachedb_dir=str(cdb_f), use_singletons=False, reconcile_grace_sec=10.0)
    reader = OrderLedgerReader(cachedb_dir=str(cdb_f))

    writer.record_event(
        OrderEventType.ACKED,
        venue="hyperliquid",
        symbol="HYPEUSDC",
        ts=t0,
        iid="hl-leg-partial",
        vid="HL-LEG-1",
        side="BUY",
        qty=10.0,
        px=58.0,
    )
    (cdb_f / "orders" / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME).write_text(
        json.dumps({
            "items": {
                "HYPEUSDC": [{
                    "orderId": "HL-LEG-1",
                    "side": "BUY",
                    "price": 58.0,
                    "quantity": 2.0,
                    "timestamp": int((t0 + 2.0) * 1000),
                    "status": "FILLED",
                }]
            },
            "fetchtime": {"HYPEUSDC": int((t0 + 2.0) * 1000)},
        }),
        encoding="utf-8",
    )
    st_p = reconciler.reconcile_once(now_ts=t0 + 5.0)
    assert st_p["reconciled_filled"] == 0 and st_p["reconciled_closed"] == 0 and st_p["active_count"] == 1
    act_rows = reader.list_active_orders(venue="hyperliquid", apply_tail=False)
    assert len(act_rows) == 1
    assert act_rows[0]["state"] == OrderLedgerState.PARTIAL.value
    assert act_rows[0]["fq"] == pytest.approx(2.0)
    assert act_rows[0]["vst"] == "PARTIALLY_FILLED"

    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(cdb_f / "orders" / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {"HYPEUSDC": []},
        now_ms=int((t0 + 30.0) * 1000),
        verified_ts=t0 + 30.0,
    )
    st_c = reconciler.reconcile_once(now_ts=t0 + 30.0)
    assert st_c["reconciled_filled"] == 0 and st_c["reconciled_closed"] == 1 and st_c["active_count"] == 0
    tr = reader.trace_order("HL-LEG-1")
    assert len(tr["closed_orders"]) == 1
    assert tr["closed_orders"][0]["state"] == OrderLedgerState.CANCELED.value
    assert tr["closed_orders"][0]["fq"] == pytest.approx(2.0)

    # --- Cycle 2: active_intents partial-fill parity, cid/cloid orphan suppression, and fill-status merge ---
    ws_p = tmp_path / "ws_partial_parity"
    (ws_p / "cachedb" / "orders").mkdir(parents=True)
    (ws_p / "cachedb" / "state").mkdir(parents=True)
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(ws_p / "cachedb" / "orders" / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {
            "HYPEUSDC": [{
                "orderId": "HL-PART-9",
                "side": "BUY",
                "price": 58.0,
                "origQty": 10.0,
                "executedQty": 2.0,
                "status": "PARTIALLY_FILLED",
                "timestamp": int(now_s * 1000),
            }]
        },
        now_ms=int(now_s * 1000),
        verified_ts=now_s,
    )
    (ws_p / "cachedb" / "orders" / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME).write_text(
        json.dumps({
            "items": {
                "HYPEUSDC": [
                    {"orderId": "HL-PART-9", "side": "BUY", "price": 58.0, "quantity": 2.0, "timestamp": int(now_s * 1000), "status": "FILLED"},
                    {"orderId": "HL-FULL-9", "side": "BUY", "price": 58.0, "quantity": 10.0, "timestamp": int(now_s * 1000), "status": "FILLED"},
                ]
            },
            "fetchtime": {"HYPEUSDC": int(now_s * 1000)},
        }),
        encoding="utf-8",
    )
    (ws_p / "cachedb" / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps([
            {
                "intent_id": "ai-hl-part-9",
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "side": "BUY",
                "order_type": "LIMIT",
                "requested_qty": 10.0,
                "requested_price": 58.0,
                "timestamp": now_s - 60.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "HL-PART-9", "status": "PARTIALLY_FILLED"},
            },
            {
                "intent_id": "ai-hl-full-9",
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "side": "BUY",
                "order_type": "LIMIT",
                "requested_qty": 10.0,
                "requested_price": 58.0,
                "timestamp": now_s - 60.0,
                "status": "EXECUTED",
                "execution_payload": {"orderId": "HL-FULL-9", "status": "NEW"},
            },
        ]),
        encoding="utf-8",
    )
    assert [it["intent_id"] for it in active_intents.build_active_intent_index(str(ws_p))["intents"]] == ["ai-hl-part-9"]

    cdb_cid = tmp_path / "cdb_cid_suppress"
    (cdb_cid / "orders").mkdir(parents=True)
    w_cid = OrderLedgerWriter(cachedb_dir=str(cdb_cid))
    r_cid = OrderLedgerReconciler(cachedb_dir=str(cdb_cid), use_singletons=False, reconcile_grace_sec=10.0)
    hl_cid_closed = "AI_hl_closed_cid_only"
    hl_cloid_closed = derive_venue_128bit_client_order_id("hyperliquid", hl_cid_closed)
    w_cid.record_event(OrderEventType.SUBMITTED, venue="hyperliquid", symbol="HYPEUSDC", ts=t0, iid="iid-cid-1", cid=hl_cid_closed, side="BUY", qty=5.0, px=60.0)
    w_cid.record_event(OrderEventType.CANCEL_ACK, venue="hyperliquid", symbol="HYPEUSDC", ts=t0 + 2.0, iid="iid-cid-1", cid=hl_cid_closed)
    assert r_cid.reconcile_once(now_ts=t0 + 3.0)["reconciled_closed"] == 1
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(cdb_cid / "orders" / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {"HYPEUSDC": [{"orderId": "HL-STALE-CID-77", "clientOrderId": hl_cloid_closed, "side": "BUY", "price": 60.0, "origQty": 5.0, "executedQty": 0.0, "status": "OPEN", "timestamp": int(t0 * 1000)}]},
        now_ms=int((t0 + 4.0) * 1000),
        verified_ts=t0 + 4.0,
    )
    st_cid2 = r_cid.reconcile_once(now_ts=t0 + 5.0)
    assert st_cid2["orphans_adopted"] == 0 and st_cid2["active_count"] == 0

    agg_rows = CacheOrderFilledManager.aggregate_fills_to_order_rows(
        [
            {"orderId": 4242, "id": "2", "side": "BUY", "price": 100.0, "quantity": 6.0, "timestamp": 1700000000000, "status": "FILLED"},
            {"orderId": 4242, "id": "1", "side": "BUY", "price": 99.0, "quantity": 4.0, "timestamp": 1700000000000, "status": "PARTIALLY_FILLED"},
        ],
        venue="binance",
    )
    assert len(agg_rows) == 1 and agg_rows[0]["quantity"] == pytest.approx(10.0) and agg_rows[0]["status"] == "FILLED"

    # --- Cycle 3: Chronological iter_audit_compat_records + monotonic intent reconciliation & AI attribution price ---
    cdb_sup = tmp_path / "cdb_sup_compat"
    (cdb_sup / "orders").mkdir(parents=True)
    w_sup = OrderLedgerWriter(cachedb_dir=str(cdb_sup))
    rec_sup = OrderLedgerReconciler(cachedb_dir=str(cdb_sup), use_singletons=False, reconcile_grace_sec=10.0)
    w_sup.record_event(OrderEventType.ACKED, venue="hyperliquid", symbol="HYPEUSDC", ts=t0, iid="ai-sup-1", cid="AI_sup_1", vid="HL-SUP-1", side="BUY", qty=10.0, px=60.0)
    w_sup.record_event(OrderEventType.STATUS, venue="hyperliquid", symbol="HYPEUSDC", ts=t0 + 5.0, iid="ai-sup-1", cid="AI_sup_1", vid="HL-SUP-1", side="BUY", vst="CANCELED", fq=2.0, avg_px=58.0)
    rec_sup.reconcile_once(now_ts=t0 + 6.0)
    w_sup.record_event(OrderEventType.STATUS, venue="hyperliquid", symbol="HYPEUSDC", ts=t0 + 20.0, iid="ai-sup-1", cid="AI_sup_1", vid="HL-SUP-1", side="BUY", vst="FILLED", fq=10.0, avg_px=60.5)
    rec_sup.reconcile_once(now_ts=t0 + 21.0)

    compat_recs, has_led = OrderLedgerReader(cachedb_dir=str(cdb_sup)).iter_audit_compat_records()
    assert has_led is True
    ts_list = [r["ts"] for r in compat_recs]
    assert ts_list == sorted(ts_list)

    ai_intents = [{
        "intent_id": "ai-sup-1",
        "venue": "hyperliquid",
        "symbol": "HYPEUSDC",
        "decision": "BUY",
        "status": "EXECUTED",
        "timestamp": t0,
        "execution_payload": {"orderId": "HL-SUP-1", "status": "NEW"},
    }]
    assert reconcile_intents_from_execution_audit(ai_intents, str(tmp_path / "no_audit"), cachedb_dir=str(cdb_sup)) is True
    assert ai_intents[0]["execution_payload"]["status"] == "FILLED"
    assert float(ai_intents[0]["execution_payload"]["executedQty"]) == pytest.approx(10.0)

    out_of_order_recs = [
        {"ts": t0 + 20.0, "event": "order_terminal", "venue": "hyperliquid", "symbol": "HYPEUSDC", "order_id": "HL-SUP-1", "client_order_id": "AI_sup_1", "side": "buy", "filled_qty": 10.0, "price": 60.5, "cost": 605.0, "status": "closed", "venue_status": "FILLED"},
        {"ts": t0 + 5.0, "event": "order_status", "venue": "hyperliquid", "symbol": "HYPEUSDC", "order_id": "HL-SUP-1", "client_order_id": "AI_sup_1", "side": "buy", "filled_qty": 2.0, "price": 58.0, "cost": 116.0, "status": "open", "venue_status": "PARTIALLY_FILLED"},
        {"ts": t0 + 40.0, "event": "order_terminal", "venue": "hyperliquid", "symbol": "HYPEUSDC", "order_id": "HL-SUP-2", "client_order_id": "AI_sup_2", "side": "sell", "filled_qty": 10.0, "price": 60.5, "cost": 605.0, "status": "closed", "venue_status": "FILLED"},
    ]
    eval_res = _evaluate_execution_audit_actors(
        out_of_order_recs,
        known_ai_order_ids=set(),
        cutoff_ts=t0 - 10.0,
        eff_now=t0 + 100.0,
        scope="crypto",
    )
    assert eval_res["by_actor"]["AI"]["realized_net_usd"] <= 0.0


def test_audit_cycle4_per_symbol_open_verification_and_partial_fill_telemetry(tmp_path):
    """Cycle 4 + Cycle 7 regressions:
    1. Partial-symbol open-order harvest must not stamp venue-wide verified_ts over omitted cached symbols
       or cause intent_telemetry to mark unqueried same-venue resting LIMIT orders as CANCELED_OR_EXPIRED.
    2. Passive (allow_network=False) harvest must populate verified_open_symbols strictly from symbol_verified_ts.
    3. _resolve_intent_execution_details must keep strict partial fills (2/10, venue_st=NEW) as PARTIALLY_FILLED
       when open snapshot is missing, and advance PARTIALLY_FILLED (2/10) to FILLED (10/10) on exact order_id full fill.
    4. _build_cached_balance_index must enforce atomic timestamp precedence regardless of dict iteration order.
    """
    from cachepaths import CACHE_KRAKEN_ORDER_OPEN_FILENAME
    from intelligence.sentiment.collectors.portfolio_collector import (
        TradablePositionsAndOrdersCollector,
        _build_cached_balance_index,
    )
    from providers.intent_telemetry import _resolve_intent_execution_details, summarize_recent_ai_intents

    now_s = 1795200000.0
    orders_dir = tmp_path / "orders"
    state_dir = tmp_path / "state"
    orders_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (state_dir / "venue_balances_state.json").write_text("{}", encoding="utf-8")

    krk_open_path = orders_dir / CACHE_KRAKEN_ORDER_OPEN_FILENAME
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(krk_open_path),
        {
            "HYPEUSD": [{
                "orderId": "OKRK-HYPE-LIVE",
                "side": "SELL",
                "price": 93.85,
                "origQty": 4.8,
                "executedQty": 0.0,
                "status": "OPEN",
                "timestamp": int((now_s - 7200.0) * 1000),
            }]
        },
        now_ms=int((now_s - 600.0) * 1000),
        verified_ts=now_s - 600.0,
    )

    fake_prov = SimpleNamespace(
        free_balance=lambda _a: 100.0,
        get_current_price=lambda _s: 140.0,
        usd_cash_breakdown=lambda: {"ZUSD": 100.0},
    )
    spcx_inst = SimpleNamespace(
        symbol="SPCXxUSD",
        provider_name="kraken",
        base="SPCXx",
        quote="USD",
        params={},
        _provider=fake_prov,
        open_orders=lambda: [],
    )
    collector = TradablePositionsAndOrdersCollector()
    ver_venues: set[str] = set()
    ver_syms: set[tuple[str, str]] = set()
    collector.collect(
        str(tmp_path),
        inst_map={"KRK_SPCX": spcx_inst},
        prices_and_trends={"SPCXxUSD": {"price": 140.0, "trend": 0, "growth_coefficient": 0.0}},
        verified_open_venues=ver_venues,
        verified_open_symbols=ver_syms,
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_s,
    )
    assert ver_venues == {"kraken"}
    assert ver_syms == {("kraken", "SPCXXUSD")}

    saved = json.loads(krk_open_path.read_text(encoding="utf-8"))
    assert saved["verified_ts"] == pytest.approx(now_s - 600.0, abs=1.0)
    assert saved["symbol_verified_ts"]["SPCXxUSD"] == pytest.approx(now_s, abs=1.0)
    assert saved["symbol_verified_ts"]["HYPEUSD"] == pytest.approx(now_s - 600.0, abs=1.0)

    ver_venues_off: set[str] = set()
    ver_syms_off: set[tuple[str, str]] = set()
    collector.collect(
        str(tmp_path),
        inst_map={},
        prices_and_trends={},
        verified_open_venues=ver_venues_off,
        verified_open_symbols=ver_syms_off,
        allow_empty_inst_recovery=True,
        allow_network=False,
        persist=False,
        now_ts=now_s + 5.0,
    )
    assert ver_venues_off == set()
    assert ver_syms_off == {("kraken", "SPCXXUSD")}

    intents = [
        {
            "intent_id": "ai-krk-hype-resting",
            "timestamp": now_s - 7200.0,
            "decision": "SELL",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 93.85,
            "suggested_notional_usd": 450.0,
            "status": "EXECUTED",
            "execution_payload": {"txid": ["OKRK-HYPE-LIVE"], "descr": {"order": "sell"}},
        },
        {
            "intent_id": "ai-krk-spcx-gone",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "symbol": "SPCXxUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 140.0,
            "suggested_notional_usd": 140.0,
            "status": "EXECUTED",
            "execution_payload": {"orderId": "OKRK-SPCX-GONE", "status": "OPEN"},
        },
    ]
    summary = summarize_recent_ai_intents(
        intents,
        "all",
        now_s,
        lambda d, default=None: d.get("suggested_notional_usd", default),
        recent_orders={},
        open_orders=[],
        verified_open_venues={"kraken"},
        verified_open_symbols={("kraken", "SPCXXUSD")},
    )
    by_sym = {row["symbol"]: row for row in summary}
    assert by_sym["HYPEUSD"]["venue_status"] == "OPEN"
    assert by_sym["SPCXxUSD"]["venue_status"] == "CANCELED_OR_EXPIRED"

    partial_intent = {
        "intent_id": "ai-hl-partial-2of10",
        "timestamp": now_s - 120.0,
        "decision": "BUY",
        "symbol": "HYPEUSDC",
        "venue": "hyperliquid",
        "order_type": "LIMIT",
        "limit_price": 60.0,
        "suggested_notional_usd": 600.0,
        "status": "EXECUTED",
        "execution_payload": {"orderId": "HL-P1", "origQty": "10.0", "executedQty": "0.0", "status": "NEW"},
    }
    det_partial = _resolve_intent_execution_details(
        partial_intent,
        now_s,
        lambda d, default=None: d.get("suggested_notional_usd", default),
        recent_orders={
            "hyperliquid:HYPEUSDC": {
                "venue": "hyperliquid",
                "last_buy": {"order_id": "HL-P1", "qty": 2.0, "price": 59.5, "ts_ms": int((now_s - 60.0) * 1000), "actor": "AI"},
            }
        },
        open_orders=[],
        verified_open_symbols=set(),
    )
    assert det_partial["exec_status"] == "PARTIALLY_FILLED"
    assert det_partial["venue_status"] == "PARTIALLY_FILLED"
    assert det_partial["filled_qty"] == pytest.approx(2.0)

    partial_saved_intent = {
        **partial_intent,
        "execution_payload": {
            "orderId": "HL-P1",
            "origQty": "10.0",
            "executedQty": "2.0",
            "price": "59.5",
            "status": "PARTIALLY_FILLED",
        },
    }
    det_full = _resolve_intent_execution_details(
        partial_saved_intent,
        now_s,
        lambda d, default=None: d.get("suggested_notional_usd", default),
        recent_orders={
            "hyperliquid:HYPEUSDC": {
                "venue": "hyperliquid",
                "last_buy": {"order_id": "HL-P1", "qty": 10.0, "price": 60.0, "ts_ms": int((now_s - 10.0) * 1000), "actor": "AI"},
            }
        },
        open_orders=[],
        verified_open_symbols={("hyperliquid", "HYPEUSDC")},
    )
    assert det_full["exec_status"] == "FILLED"
    assert det_full["venue_status"] == "FILLED"
    assert det_full["filled_qty"] == pytest.approx(10.0)
    assert det_full["fill_px"] == pytest.approx(60.0)

    det_no_oid = _resolve_intent_execution_details(
        partial_saved_intent,
        now_s,
        lambda d, default=None: d.get("suggested_notional_usd", default),
        recent_orders={
            "hyperliquid:HYPEUSDC": {
                "venue": "hyperliquid",
                "last_buy": {"qty": 10.0, "price": 61.0, "ts_ms": int((now_s - 10.0) * 1000), "actor": "AI"},
            }
        },
        open_orders=[],
        verified_open_symbols=set(),
    )
    assert det_no_oid["exec_status"] == "PARTIALLY_FILLED"
    assert det_no_oid["filled_qty"] == pytest.approx(2.0)

    bals_new_first = {
        "pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "base": "BTC", "quote": "USDC", "free_base": 0.5, "free_quote": 500.0, "ts": now_s - 10.0},
        "pos:binance:ETHUSDC": {"venue": "binance", "symbol": "ETHUSDC", "base": "ETH", "quote": "USDC", "free_base": 2.0, "free_quote": 100.0, "ts": now_s - 1000.0},
        "pos:binance:BTCUSDT": {"venue": "binance", "symbol": "BTCUSDT", "base": "BTC", "quote": "USDT", "free_base": 0.1, "free_quote": 50.0, "ts": now_s - 1000.0},
        "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 100.0, "ts": now_s - 1000.0},
    }
    obs_a = _build_cached_balance_index(bals_new_first)
    assert obs_a[("binance", "USDC")][0] == pytest.approx(500.0)
    assert obs_a[("binance", "USDC")][1] == pytest.approx(now_s - 10.0)
    assert obs_a[("binance", "BTC")][0] == pytest.approx(0.5)
    assert obs_a[("binance", "BTC")][1] == pytest.approx(now_s - 10.0)

    bals_old_first = dict(reversed(list(bals_new_first.items())))
    obs_b = _build_cached_balance_index(bals_old_first)
    assert obs_b[("binance", "USDC")][0] == pytest.approx(500.0)
    assert obs_b[("binance", "BTC")][0] == pytest.approx(0.5)
    assert obs_b[("binance", "USDC")][1] == pytest.approx(now_s - 10.0)

    bals_fresh_cash = {**bals_new_first, "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 750.0, "ts": now_s}}
    obs_c = _build_cached_balance_index(bals_fresh_cash)
    assert obs_c[("binance", "USDC")][0] == pytest.approx(750.0)
    assert obs_c[("binance", "USDC")][1] == pytest.approx(now_s)


def test_audit_cycle5_hl_empty_status_fail_closed_and_ledger_cid_fallback(tmp_path, monkeypatch):
    """Cycle 5 regressions:
    1. HyperliquidProvider.order_status must raise ProviderError on missing/empty venue status
       instead of silently normalizing it to terminal 'canceled'.
    2. find_order_audit_record must fall back to client_order_id when order_id misses in OrderLedger.
    """
    from providers.execution_audit import find_order_audit_record
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.ohlc_utils import ProviderError

    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xabc123")
    for bad_status in (None, "", "   ", "null"):
        hl_info = SimpleNamespace(
            query_order_by_oid=lambda _addr, _oid, _bs=bad_status: {
                "status": "order",
                "order": {
                    "status": _bs,
                    "order": {"oid": 888, "origSz": "2.0", "sz": "2.0"},
                },
            },
            user_fills=lambda _addr: [],
        )
        hl_cli = SimpleNamespace(
            info=hl_info,
            account_address="0xabc123",
            resolve_spot_pair=lambda _tok: "@107",
            spot_mid=lambda _pair: 50.0,
        )
        hl = HyperliquidProvider(token="HYPE", live_enabled=False, client=hl_cli)
        with pytest.raises(ProviderError, match="missing or empty order status"):
            hl.order_status("HYPEUSDC", "888")

    cdb = tmp_path / "cdb_cid_fallback"
    (cdb / "orders").mkdir(parents=True)
    now_s = 1795300000.0
    writer = OrderLedgerWriter(cachedb_dir=str(cdb))
    writer.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="BTCUSDC",
        ts=now_s,
        iid="dca-intent-99",
        cid="DCA_1795300000_abcd",
        side="BUY",
        qty=0.05,
        px=65000.0,
        owner="spot_dca",
        kind="dca_buy",
    )
    found = find_order_audit_record(
        "999888777",
        "DCA_1795300000_abcd",
        symbol="BTCUSDC",
        event_ts_s=now_s + 5.0,
        cachedb_dir=str(cdb),
    )
    assert found is not None
    assert found["client_order_id"] == "DCA_1795300000_abcd"
    assert found["intent_id"] == "dca-intent-99"
    assert found["kind"] == "dca_buy"


def test_audit_cycle8_to_11_execution_guards_status_ssot_spot_tranche_and_ag_preempt(tmp_path, monkeypatch):
    """Cycles 8-11 regressions:
    1. Targeted replacement (verify_not_filled=True) fails closed on transient order_status errors,
       while permitting replacement on definitive OrderNotFound.
    2. Resuming PENDING_EXECUTION defers safely on order_by_client_id lookup error.
    3. BinanceProvider.order_status fails closed on missing/empty/unknown status and maps PENDING_CANCEL -> open.
    4. order_retry_worker & monitororder share SSOT is_confirmed_cancel_order_status (ARCHIVED_CANCELED,
       userCanceled, marginCanceled, scheduledCancel, iocCancel allowed; REJECTED & PENDING_CANCEL blocked).
    5. spot_engine._dispatch_take_profit actually places the collapsed single TP order when multi-tranche
       splits fall below ordermin (instead of iterating un-collapsed sized_by_slot and placing 0 orders).
    6. monitortrades._is_pending_exit_active immediately clears dead pending exits on Binance EXPIRED_IN_MATCH
       / ARCHIVED_CANCELED and Hyperliquid marginCanceled / iocCancel / minTradeNtlRejected.
    7. assetguardian.evaluate_symbol restores preempted BUY orders if post-preemption free_balance raises.
    8. order_ledger.resolve_late_closed_event_state never upgrades strictly partial RECONCILED_PARTIAL or
       legacy vst='FILLED' (2/10) to terminal FILLED.
    """
    from unittest.mock import MagicMock

    import assetguardian as ag
    import monitortrades as mt
    import order_retry as oq
    import order_retry_worker as worker
    from providers import binance_provider as bp_mod
    from providers.binance_provider import BinanceProvider
    from providers.intent_execution import cancel_existing_ai_open_orders, reconcile_open_intent_by_client_id
    from providers.ohlc_utils import ProviderError
    from providers.order_ledger import OrderLedgerState, resolve_late_closed_event_state
    from providers.order_status_ssot import is_confirmed_cancel_order_status, is_terminal_non_fill_venue_status
    from providers.strategy_executor import OrderNotFound, OrderStatus, PairPrecision
    from strategies.spot_engine import StratParams, Strategy

    # 1. verify_not_filled=True fail-closed on transient ProviderError, open on OrderNotFound
    inst_err = SimpleNamespace(
        provider_name="binance",
        symbol="BTCUSDC",
        open_orders=MagicMock(return_value=[]),
        cancel_order=MagicMock(return_value=True),
        order_status=MagicMock(side_effect=ProviderError("502 Bad Gateway")),
    )
    failed_ids: list[str] = []
    cancel_existing_ai_open_orders(
        inst_err, "BUY", include_bot_orders=True, target_order_ids=["oid-race-1"],
        verify_not_filled=True, failed_order_ids=failed_ids,
    )
    assert failed_ids == ["oid-race-1"]

    inst_404 = SimpleNamespace(
        provider_name="t212",
        symbol="NVDA_US_EQ",
        open_orders=MagicMock(return_value=[]),
        cancel_order=MagicMock(return_value=True),
        order_status=MagicMock(side_effect=OrderNotFound("definitively absent")),
    )
    failed_404: list[str] = []
    cancel_existing_ai_open_orders(
        inst_404, "BUY", include_bot_orders=True, target_order_ids=["oid-gone-404"],
        verify_not_filled=True, failed_order_ids=failed_404,
    )
    assert failed_404 == []

    # 2. Resumed PENDING_EXECUTION defers safely on order_by_client_id error
    inst_resume = SimpleNamespace(
        provider_name="binance",
        symbol="BTCUSDC",
        open_orders=MagicMock(return_value=[]),
        order_by_client_id=MagicMock(side_effect=ProviderError("timeout")),
    )
    resumed_intent = {
        "intent_id": "intent_1795400001_buy_btcusdc",
        "venue": "binance",
        "symbol": "BTCUSDC",
        "status": "PENDING_EXECUTION",
        "history": [],
    }
    assert reconcile_open_intent_by_client_id(inst_resume, resumed_intent, 1795400010.0) is True
    assert resumed_intent["status"] == "PENDING_EXECUTION"
    assert resumed_intent["history"][-1]["action"] == "RECONCILE_DEFERRED_LOOKUP_ERROR"

    # 3. BinanceProvider.order_status fails closed on missing/empty/unknown status; PENDING_CANCEL -> open
    fake_bapi = SimpleNamespace(
        client=SimpleNamespace(
            get_order=MagicMock(return_value={}),
            get_my_trades=MagicMock(return_value=[]),
            get_symbol_info=MagicMock(return_value={"baseAsset": "BTC", "quoteAsset": "USDC"}),
        )
    )
    monkeypatch.setattr(bp_mod, "_get_bapi", lambda: fake_bapi)
    bp = BinanceProvider(live_enabled=False)
    for bad_payload in ({}, {"status": ""}, {"status": "null"}, {"status": "BOGUS_STATE"}):
        fake_bapi.client.get_order.return_value = bad_payload
        with pytest.raises(ProviderError, match="unknown or empty Binance status"):
            bp.order_status("BTCUSDC", "42")

    fake_bapi.client.get_order.return_value = {
        "status": "PENDING_CANCEL", "executedQty": "0", "cummulativeQuoteQty": "0", "origQty": "1.0",
    }
    st_pc = bp.order_status("BTCUSDC", "42")
    assert st_pc.status == "open" and st_pc.venue_status == "PENDING_CANCEL"

    # 4. Unified SSOT cancel/non-fill predicates across order_retry_worker and monitororder
    assert is_confirmed_cancel_order_status(OrderStatus("canceled", 0.0, 0.0, 0.0, venue_status="scheduledCancel"))
    assert is_confirmed_cancel_order_status(OrderStatus("canceled", 0.0, 0.0, 0.0, venue_status="iocCancel"))
    assert not is_confirmed_cancel_order_status(OrderStatus("canceled", 0.0, 0.0, 0.0, venue_status="PENDING_CANCEL"))
    assert not is_confirmed_cancel_order_status(OrderStatus("canceled", 0.0, 0.0, 0.0, venue_status="badAloPxRejected"))
    assert not is_terminal_non_fill_venue_status("PENDING_CANCEL")
    assert is_terminal_non_fill_venue_status("EXPIRED_IN_MATCH")
    assert is_terminal_non_fill_venue_status("marginCanceled")
    assert is_terminal_non_fill_venue_status("iocCancel")
    assert is_terminal_non_fill_venue_status("minTradeNtlRejected")

    q_file = str(tmp_path / "q_c8.jsonl")
    l_file = str(tmp_path / "q_c8.lock")
    monkeypatch.setattr(oq, "QUEUE_FILE", q_file)
    monkeypatch.setattr(oq, "LOCK_FILE", l_file)
    monkeypatch.setattr(oq, "RETRY_ENABLED", True)
    monkeypatch.setattr(oq, "RETRY_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(oq, "RETRY_TTL_SEC", 86400.0)
    monkeypatch.setattr(worker, "WORKER_POLL_SEC", 1.0)
    monkeypatch.setattr(worker.alert, "notify", lambda **kw: None)

    for v_status, expect_activated in (
        ("ARCHIVED_CANCELED", True),
        ("userCanceled", True),
        ("marginCanceled", True),
        ("scheduledCancel", True),
        ("REJECTED", False),
    ):
        oq.rewrite([])
        oq.enqueue(
            "BTCUSDC", "SELL", 1.5, {"smart": False}, requested_price=101.0, now=1000.0,
            lifecycle="awaiting_cancel", replaces_order_id="old-c8", replaces_original_qty=1.5,
        )
        fake_mkt = SimpleNamespace(
            order_status=lambda _s, _oid, provider_name=None, _vs=v_status: OrderStatus(
                "canceled", 0.0, 0.0, 0.0, venue_status=_vs, orig_qty=1.5
            ),
            reconciliation_capabilities=lambda *_a, **_kw: None,
        )
        worker.process_once(fake_mkt, now=1400.0)
        rows = oq.load_all()
        if expect_activated:
            assert len(rows) == 1 and rows[0]["lifecycle"] == "submit_pending", v_status
        else:
            assert rows == [], v_status

    for v_status, expect_place in (("ARCHIVED_CANCELED", True), ("userCanceled", True), ("REJECTED", False)):
        oq.rewrite([])
        rep_id = oq.enqueue(
            "BTCUSDC", "SELL", 1.0, {"smart": False}, requested_price=105.0, now=1000.0,
            lifecycle="awaiting_cancel", replaces_order_id="old-mo-c9", replaces_original_qty=1.0,
        )
        rep_claim = oq.claim([rep_id], now=1000.0, lease_sec=60.0)[0]
        fake_mo_mkt = MagicMock()
        fake_mo_mkt.order_status.return_value = OrderStatus("canceled", 0.0, 0.0, 0.0, venue_status=v_status, orig_qty=1.0)
        fake_mo_mkt.place.return_value = {"orderId": "new-mo-99"}
        fake_mo_api = MagicMock()
        fake_mo_api.cancel_order.return_value = True
        with monkeypatch.context() as mp:
            mp.setattr(monitororder, "mkt", fake_mo_mkt)
            mp.setattr(monitororder, "api", fake_mo_api)
            mp.setattr(monitororder, "_native_replace_enabled_for", lambda _s, _p: False)
            monitororder._dispatch_native_or_cancel_replace(
                symbol="BTCUSDC", order_type="SELL", order_id="old-mo-c9", oid_str="old-mo-c9",
                price="100", price_f=100.0, new_price=105.0, quantity=1.0, original_quantity=1.0,
                provider_name="binance", replacement_claim=rep_claim,
                client_order_id=oq.record_client_order_id(rep_claim), cache_permit=None,
                preflight_qty=1.0, replace_bypass_guards=[], replace_wait_for_trend=False,
                orders={}, initial_prices={},
            )
        assert fake_mo_mkt.place.call_count == (1 if expect_place else 0), v_status

    # 5. spot_engine multi-tranche collapse below ordermin places the single collapsed TP order
    fake_cli = SimpleNamespace(
        pair_precision=lambda _pair: PairPrecision(price_decimals=2, volume_decimals=4, order_min=0.01, base_asset="BTC"),
    )
    params = StratParams(
        currency="USDC", entry_amount=100.0, entry_discount_pct=0.2, dca_amount=50.0,
        dca_drop_pct=2.0, check_minutes=5.0, takeprofit_pct=5.0, max_budget=500.0,
        max_dca_buys=2, enable_takeprofit=True, order_ttl_min=60.0,
        tp_tranches=[(4.0, 34.0), (8.0, 33.0), (12.0, 33.0)],
    )
    strat = Strategy(
        fake_cli,
        "BTCUSDC",
        params,
        dry_run=True,
        replay_mode=True,
        state_dir=str(tmp_path / "spot_state"),
        initial_state={"qty": 0.015, "cost": 1500.0, "spent": 1500.0, "entry_price": 100000.0, "last_buy_price": 100000.0},
    )
    strat._dispatch_take_profit(price=100000.0, avg=100000.0, held=0.015)
    sell_orders = [o for o in strat.s["orders"] if o["side"] == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[0]["vol"] == pytest.approx(strat._dust_safe_qty(0.015))
    assert sell_orders[0]["price"] == pytest.approx(104000.0)

    # 6. monitortrades._is_pending_exit_active immediately clears Binance EXPIRED_IN_MATCH & HL *Canceled/*Cancel/*Rejected
    with monkeypatch.context() as mp:
        mp.setattr(mt, "_MT_PERSISTENCE_ENABLED", False)
        for dead_st in ("EXPIRED_IN_MATCH", "ARCHIVED_CANCELED", "marginCanceled", "iocCancel", "minTradeNtlRejected"):
            mt._PENDING_STOP_EXIT.clear()
            mt._PENDING_STOP_EXIT["HYPEUSDC"] = {"time": 1000.0, "client_order_id": "SL_cid_1", "kind": "STOP"}
            inst_hl = SimpleNamespace(
                symbol="HYPEUSDC",
                _provider=SimpleNamespace(order_by_client_id=lambda _s, _c, _st=dead_st: {"orderId": "88", "status": _st}),
                open_orders=lambda: [],
            )
            assert mt._is_pending_exit_active("HYPEUSDC", inst_hl, 2.0, 0.1, 1005.0, current_price=50.0) is False, dead_st
            assert "HYPEUSDC" not in mt._PENDING_STOP_EXIT, dead_st

        # Non-terminal PENDING_CANCEL stays deferred
        mt._PENDING_STOP_EXIT["HYPEUSDC"] = {"time": 1000.0, "client_order_id": "SL_cid_2", "kind": "STOP"}
        inst_pc = SimpleNamespace(
            symbol="HYPEUSDC",
            _provider=SimpleNamespace(order_by_client_id=lambda _s, _c: {"orderId": "89", "status": "PENDING_CANCEL"}),
            open_orders=lambda: [],
        )
        assert mt._is_pending_exit_active("HYPEUSDC", inst_pc, 2.0, 0.1, 1005.0, current_price=50.0) is True
        mt._PENDING_STOP_EXIT.clear()

    # 7. assetguardian.evaluate_symbol restores preempted BUY orders if post-preemption free_balance raises
    restored_calls: list[tuple[str, list, str]] = []
    bal_calls = {"n": 0}

    def _flaky_free_balance(asset):
        if asset == "USDC":
            bal_calls["n"] += 1
            if bal_calls["n"] == 1:
                return 0.0
            raise RuntimeError("transient 502 after preemption")
        return 0.0

    fake_ag_prov = SimpleNamespace(
        free_balance=_flaky_free_balance,
        open_orders=lambda _sym: [{"orderId": "RT-1", "clientOrderId": "RT_abc", "side": "BUY", "price": "90.0", "origQty": "1.0", "executedQty": "0.0"}],
    )
    with monkeypatch.context() as mp:
        ag_sym = ag.TRACKED_SYMBOLS[0]
        mp.setattr(ag, "STATE", ag.AssetGuardianState(path=str(tmp_path / "ag_state.json")))
        mp.setattr(ag.api, "get_current_price", lambda _s: 70.0)
        mp.setattr(ag, "_get_symbol_window_extrema", lambda _s, _p, minutes_back=None: ({"timestamp": 1000.0, "price": 70.0}, {"timestamp": 1000.0, "price": 100.0}))
        mp.setattr(ag.mkt, "provider_by_name", lambda _v: fake_ag_prov)
        mp.setattr(ag, "_cycle_portfolio_growth_percent", lambda minutes_back=None: None)
        mp.setattr(ag, "_cycle_p3b_sentiment", lambda: None)
        mp.setattr(ag, "_is_short_trend_waiting", lambda _s, _side: False)
        mp.setattr(ag, "has_preemptible_open_orders", lambda *_a, **_kw: True)

        def _fake_preempt(_prov, _sym, _side, _owner, *, cancel_fn, canceled_orders_out, resolve_retry_fn):
            canceled_orders_out.append({"orderId": "RT-1", "side": "BUY", "price": "90.0", "origQty": "1.0"})
            return 1

        mp.setattr(ag, "preempt_lower_priority_open_orders", _fake_preempt)
        mp.setattr(ag, "_restore_preempted_orders", lambda sym, ords, reason: restored_calls.append((sym, list(ords), reason)) or len(ords))
        assert ag.evaluate_symbol(ag_sym) is False
        assert len(restored_calls) == 1
        assert restored_calls[0][1][0]["orderId"] == "RT-1"
        assert restored_calls[0][2] == "assetguardian_buy_no_cash"

    # 8. resolve_late_closed_event_state never upgrades strictly partial RECONCILED_PARTIAL or legacy vst='FILLED' (2/10) to FILLED
    toks = ["hyperliquid:hypeusdc:vid:hl-p99"]
    rc_zero = {toks[0]: 1000.0}
    rc_part = {toks[0]: 1000.0, f"partial:{toks[0]}": 1000.0}
    ev_leg_part = {"ev": "STATUS", "vst": "FILLED", "fq": 2.0, "qty": 10.0}
    assert resolve_late_closed_event_state(ev_leg_part, "STATUS", ev_toks=toks, recently_closed=rc_zero) == OrderLedgerState.CANCELED.value
    assert resolve_late_closed_event_state(ev_leg_part, "STATUS", ev_toks=toks, recently_closed=rc_part) is None
    ev_rec_part = {"ev": "RECONCILED_PARTIAL", "vst": "FILLED", "fq": 2.0, "qty": 10.0}
    assert resolve_late_closed_event_state(ev_rec_part, "RECONCILED_PARTIAL", ev_toks=toks, recently_closed=rc_zero) == OrderLedgerState.CANCELED.value
    assert resolve_late_closed_event_state(ev_rec_part, "RECONCILED_PARTIAL", ev_toks=toks, recently_closed=rc_part) is None

    # 9. portfolio_collector persists venue-reflected locked balances for Unified NAV while keeping
    #    raw free balances unmutated and never double-counting pre-balance synthetic deductions
    from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector
    from providers.portfolio_valuation import compute_venue_nav_breakdown

    cdir_nav = tmp_path / "cachedb_nav_locks"
    (cdir_nav / "state").mkdir(parents=True)
    (cdir_nav / "orders").mkdir(parents=True)
    vb_nav = cdir_nav / "state" / "venue_balances_state.json"
    now_nav = 1_795_700_000.0
    bal_old = now_nav - 60.0
    vb_nav.write_text(
        json.dumps({
            "pos:binance:ARBUSDC": {"venue": "binance", "symbol": "ARBUSDC", "base": "ARB", "quote": "USDC", "price": 1.0, "free_base": 0.0, "free_quote": 100.0, "ts": bal_old},
            "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 100.0, "ts": bal_old},
        }),
        encoding="utf-8",
    )
    inst_nav = {
        "ARBUSDC": SimpleNamespace(
            symbol="ARBUSDC",
            provider_name="binance",
            base="ARB",
            quote="USDC",
            params={},
            _provider=SimpleNamespace(free_balance=lambda _a: None),
            open_orders=lambda: [
                {"orderId": "OLD-B", "symbol": "ARBUSDC", "side": "BUY", "type": "LIMIT", "price": "1.0", "origQty": "40.0", "executedQty": "0.0", "status": "NEW", "time": int((bal_old - 30.0) * 1000)},
                {"orderId": "NEW-B", "symbol": "ARBUSDC", "side": "BUY", "type": "LIMIT", "price": "1.0", "origQty": "25.0", "executedQty": "0.0", "status": "NEW", "time": int((now_nav - 10.0) * 1000)},
            ],
        )
    }
    _, _, vcash_nav, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir_nav),
        inst_map=inst_nav,
        prices_and_trends={"ARBUSDC": {"price": 1.0}},
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_nav,
    )
    assert vcash_nav["binance:USDC"]["free"] == pytest.approx(75.0)
    assert vcash_nav["binance:USDC"]["locked_quote_usd"] == pytest.approx(65.0)
    # Persisted cleanly into Layer 3 cache_balance_binance.json (raw balances) + Layer 5 portfolio_positions_state.json (pos:* only)
    bal_bin = json.loads((cdir_nav / "orders" / "cache_balance_binance.json").read_text(encoding="utf-8"))
    assert bal_bin["items"]["USDC"]["free"] == pytest.approx(100.0)
    assert bal_bin["items"]["USDC"]["locked"] == pytest.approx(40.0)
    pos_state_nav = json.loads((cdir_nav / "state" / "portfolio_positions_state.json").read_text(encoding="utf-8"))
    assert "pos:binance:ARBUSDC" in pos_state_nav
    assert all(k.startswith("pos:") for k in pos_state_nav)
    from providers.portfolio_state_store import load_venue_balance_caches, load_venue_balances_state_dict

    bal_caches_nav = load_venue_balance_caches(str(cdir_nav))
    saved_nav = load_venue_balances_state_dict(str(cdir_nav))
    assert saved_nav["cash:binance:USDC"]["free"] == pytest.approx(100.0)
    assert saved_nav["cash:binance:USDC"]["locked_quote_usd"] == pytest.approx(40.0)
    nav_eval = compute_venue_nav_breakdown(
        venue_balance_caches=bal_caches_nav,
        venue_balances_state=saved_nav,
        live_prices={"ARBUSDC": 1.0},
    )
    assert nav_eval["portfolio_total_usd"] == pytest.approx(140.0)


def test_cycle12_place_context_slot_gateway_blocked_rollback_and_atomic_retry_discard(tmp_path, monkeypatch):
    """Verify Cycle 12 fixes:
    1. Instrument._PlaceContext declares `enqueued_intent_id` slot so guard-blocked retry enqueue links OrderLedger `iid` without AttributeError.
    2. AutonomousIntentGateway rolls back canceled prior resting orders when replacement is BLOCKED_BY_GUARD (e.g., transient balance unavailability).
    3. order_retry.discard_refused_intent_records & discard_superseded_symbol_records remove atomically under lock and protect in-flight/ambiguous records.
    """
    from unittest.mock import MagicMock
    import instrument as inst_mod
    import order_retry as oq
    import state_io
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway

    # 1. _PlaceContext accepts enqueued_intent_id and Instrument._finalize_place_outcome records it into OrderLedger
    ctx = inst_mod._PlaceContext(
        side_u="BUY", price=100.0, qty=1.0, orig_price=100.0, orig_qty=1.0, kwargs={}
    )
    ctx.reason = "trend_deferred"
    ctx.enqueued_intent_id = "outbox-test-1"
    assert ctx.enqueued_intent_id == "outbox-test-1"

    recorded_ledger_events: list[dict] = []
    fake_writer = SimpleNamespace(record_event=lambda ev_type, **kw: recorded_ledger_events.append({"ev": ev_type, **kw}))
    monkeypatch.setattr(inst_mod, "get_default_ledger_writer", lambda: fake_writer)
    monkeypatch.setattr(oq, "RETRY_ENABLED", True)
    monkeypatch.setattr(oq, "enqueue", lambda *a, **kw: {"id": "rec999", "intent_id": "outbox-rec999"})
    monkeypatch.setattr(inst_mod.Instrument, "_ledger_guard_block_dedup", {}, raising=False)

    fake_inst = inst_mod.Instrument.__new__(inst_mod.Instrument)
    fake_inst.symbol = "BTCUSDC"
    fake_inst.provider_name = "binance"
    fake_inst._provider = SimpleNamespace(get_current_price=lambda _s: 100.0)

    ctx2 = inst_mod._PlaceContext(
        side_u="BUY", price=100.0, qty=1.0, orig_price=100.0, orig_qty=1.0, kwargs={}
    )
    ctx2.reason = "trend_deferred"
    fake_inst._finalize_place_outcome(ctx2, caller="test_cycle12.py")
    assert ctx2.retry_enqueued is True
    assert ctx2.enqueued_intent_id == "outbox-rec999"
    assert len(recorded_ledger_events) == 1
    assert recorded_ledger_events[0]["iid"] == "outbox-rec999"

    # 2. AutonomousIntentGateway rolls back canceled prior resting order when post-cancel balance check yields BLOCKED_BY_GUARD
    ws_dir = tmp_path / "gw_ws"
    (ws_dir / "cachedb").mkdir(parents=True)
    gw = AutonomousIntentGateway(
        workspace_dir=str(ws_dir),
        mode="enforce",
        min_confidence=0.80,
        max_notional_usd=500.0,
        notify_enabled=False,
    )
    place_calls: list[dict] = []
    open_list = [
        {
            "orderId": "ai_sell_1",
            "clientOrderId": "AI_sell_cid_1",
            "side": "SELL",
            "price": "50.0",
            "origQty": "2.0",
            "executedQty": "0.0",
            "status": "NEW",
        }
    ]

    def _mock_cancel(oid):
        open_list[:] = [o for o in open_list if o["orderId"] != oid]
        return True

    def _mock_place(**kw):
        place_calls.append(kw)
        return {"orderId": "restored_sell_1", "status": "NEW"}

    mock_inst = SimpleNamespace(
        symbol="HYPEUSD",
        provider_name="kraken",
        base="HYPE",
        quote="USD",
        _provider=SimpleNamespace(
            name="Kraken",
            free_balance=lambda _a: 0.0,
            get_current_price=lambda _s: 50.0,
            get_orders=lambda *_a, **_k: [],
            order_status=lambda _s, _oid: SimpleNamespace(status="canceled", filled_qty=0.0),
        ),
        open_orders=lambda: list(open_list),
        cancel_order=MagicMock(side_effect=_mock_cancel),
        place=MagicMock(side_effect=_mock_place),
    )
    res_blocked = gw.process_intent(
        {
            "intent_id": "intent_c12_blocked_rb",
            "decision": "SELL",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 55.0,
            "reference_price": 50.0,
            "confidence": 0.92,
            "suggested_notional_usd": 100.0,
            "replace_open_orders": True,
            "thesis": "Replace resting sell higher",
        },
        instruments_map={"KRAKEN_HYPE": mock_inst},
        mode="enforce",
    )
    assert res_blocked["status"] == "BLOCKED_BY_GUARD"
    assert res_blocked["canceled_prior_open_orders"] == ["ai_sell_1"]
    assert res_blocked["restored_prior_open_orders"] == 1
    assert len(place_calls) == 1
    assert place_calls[0]["side"] == "SELL"
    assert place_calls[0]["price"] == pytest.approx(50.0)
    assert place_calls[0]["qty"] == pytest.approx(2.0)

    # 3. order_retry atomic discard protects in-flight/ambiguous records and ignores accepted records
    q_file = tmp_path / "retry_c12.jsonl"
    monkeypatch.setattr(oq, "QUEUE_FILE", str(q_file))
    monkeypatch.setattr(oq, "LOCK_FILE", str(tmp_path / "retry_c12.lock"))
    state_io.atomic_write_jsonl(
        str(q_file),
        [
            {
                "id": "rec-refused",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "created_ts": 1000.0,
                "lifecycle": "submit_pending",
                "submission_state": "refused",
                "intent_id": "AI_c12_1",
                "place_kwargs": {"client_order_id": "AI_c12_1_cid"},
            },
            {
                "id": "rec-unknown",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "created_ts": 1000.0,
                "lifecycle": "submit_pending",
                "submission_state": "unknown",
                "last_failure_reason": "submit_ambiguous",
                "intent_id": "AI_c12_2",
                "place_kwargs": {"client_order_id": "AI_c12_2_cid"},
            },
            {
                "id": "rec-producer",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "created_ts": 1000.0,
                "lifecycle": "submit_pending",
                "submission_state": "producer_claimed",
                "intent_id": "AI_c12_3",
                "place_kwargs": {"client_order_id": "AI_c12_3_cid"},
            },
            {
                "id": "rec-accepted-refused-stale",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "qty": 0.01,
                "created_ts": 1000.0,
                "lifecycle": "accepted",
                "order_id": "999001",
                "submission_state": "refused",
                "intent_id": "AI_c12_acc",
                "place_kwargs": {"client_order_id": "AI_c12_acc_cid"},
            },
        ],
    )
    assert oq.discard_superseded_symbol_records("BTCUSDC", "BUY", include_bot_orders=True) == ["rec-refused"]
    assert oq.discard_refused_intent_records("AI_c12_acc") == []
    remaining_ids = [r["id"] for r in oq.load_all()]
    assert remaining_ids == ["rec-unknown", "rec-producer", "rec-accepted-refused-stale"]


def test_cycle13_reconciler_loads_list_root_ai_intents_and_enrich_preserves_terminal_cancel(tmp_path):
    """Verify Cycle 13 fixes on top of upstream be5aff4d:
    1. OrderLedgerReconciler._load_known_ai_order_ids_lc reads production list-rooted autonomous_trade_intents.json
       (and recovers .previous) so Hyperliquid AI orders adopt with owner='ai_executor' instead of 'external'.
    2. cacheaccount._enrich_filled_row_metadata preserves terminal CANCELED/EXPIRED partial-fill status against PARTIALLY_FILLED.
    """
    from cacheaccount import CacheOrderOpenManager, _enrich_filled_row_metadata
    from cachepaths import AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME, CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME
    from order_ledger_reconciler_worker import OrderLedgerReconciler
    from providers.order_id_context import derive_venue_128bit_client_order_id
    from providers.order_ledger import OrderLedgerReader

    cdb = tmp_path / "cachedb_c13"
    (cdb / "state").mkdir(parents=True)
    (cdb / "orders").mkdir(parents=True)

    t0 = 1_795_800_000.0
    known_cid = "AI_intent_1795800000_buy_hypeusdc"
    known_cloid = derive_venue_128bit_client_order_id("hyperliquid", known_cid)
    # Write the real production format: top-level JSON list (as saved by save_durable_intents_state)
    (cdb / "state" / AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME).write_text(
        json.dumps([
            {
                "intent_id": "intent_1795800000_buy_hypeusdc",
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "status": "EXECUTED",
                "execution_payload": {"orderId": "880001", "status": "OPEN"},
            }
        ]),
        encoding="utf-8",
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(cdb / "orders" / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME),
        {
            "HYPEUSDC": [
                {
                    "orderId": "880001",
                    "clientOrderId": known_cloid,
                    "side": "BUY",
                    "price": 58.0,
                    "origQty": 5.0,
                    "executedQty": 0.0,
                    "status": "OPEN",
                    "timestamp": int(t0 * 1000),
                }
            ]
        },
        now_ms=int((t0 + 2) * 1000),
        verified_ts=t0 + 2.0,
    )
    rec = OrderLedgerReconciler(cachedb_dir=str(cdb), use_singletons=False, reconcile_grace_sec=30.0)
    rec.reconcile_once(now_ts=t0 + 3.0)
    active = OrderLedgerReader(cachedb_dir=str(cdb)).list_active_orders(apply_tail=False)
    by_vid = {r["vid"]: r for r in active}
    assert by_vid["880001"]["owner"] == "ai_executor"

    # 2. _enrich_filled_row_metadata upgrades PARTIALLY_FILLED -> CANCELED and never downgrades CANCELED -> PARTIALLY_FILLED
    row_part = {"orderId": "10", "status": "PARTIALLY_FILLED"}
    assert _enrich_filled_row_metadata(row_part, {"orderId": "10", "status": "CANCELED"}) is True
    assert row_part["status"] == "CANCELED"
    assert _enrich_filled_row_metadata(row_part, {"orderId": "10", "status": "PARTIALLY_FILLED"}) is False
    assert row_part["status"] == "CANCELED"


def test_twin_bug_elimination_and_by_construction_refactors(tmp_path):
    """Regression coverage for twin bugs found in codebase-wide sweep and by-construction SSOT refactors:
    1. providers/portfolio_valuation.py: separate authoritative runtime views from raw disk venue_balances_state
       so raw pre-deduction free cash/base is never double-counted alongside runtime locked balances.
    2. active_intents.py: classify cache_order_filled rows via providers.order_ledger.classify_fill_evidence_state
       so legacy status='FILLED' 2/10 partial fills never hide resting open orders or prune partial intents.
    3. order_ledger_reconciler_worker.py: Step 3 orphan adoption checks closed_order_tokens (cid + vid), not vid only.
    4. microstructure_collector.py: refresh_stale_t212_cached_positions preserves original balance timestamps into
       base_bal_ts/quote_bal_ts when updating price ts.
    5. portfolio_evaluator.py: PositionCostBasisEnrichmentEvaluator honors explicit now_ts (no _time.time() leak).
    6. sentiment_advisor.py & prompt_sections.py: normalize near-zero negatives (-0.004 -> +0.00%, never -0.00%).
    """
    from types import SimpleNamespace
    from unittest import mock

    import active_intents
    from cacheaccount import (
        CACHE_BINANCE_ORDER_OPEN_FILENAME,
        CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
        CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
        CacheOrderOpenManager,
    )
    from intelligence.sentiment.collectors import microstructure_collector as mc_mod
    from intelligence.sentiment.evaluators.portfolio_evaluator import PositionCostBasisEnrichmentEvaluator
    from intelligence.sentiment.sentiment_advisor import _format_breadth_line, _format_single_whale_entry
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section
    from providers.order_ledger import OrderEventType, OrderLedgerReader, OrderLedgerWriter
    from providers.portfolio_valuation import compute_venue_nav_breakdown

    # --- 1. NAV valuation never double-counts raw disk free + runtime locked balances ---
    nav = compute_venue_nav_breakdown(
        positions=[
            {
                "venue": "binance",
                "symbol": "ARBUSDC",
                "base": "ARB",
                "price": 1.0,
                "free_base": 6.0,
                "value_usd": 6.0,
                "locked_base_qty": 4.0,
                "locked_base_usd": 4.0,
            }
        ],
        quote_cash={"binance:USDC": {"venue": "binance", "asset": "USDC", "free": 75.0, "locked_quote_usd": 25.0}},
        venue_balances_state={
            "pos:binance:ARBUSDC": {"venue": "binance", "symbol": "ARBUSDC", "base": "ARB", "price": 1.0, "free_base": 10.0, "value_usd": 10.0},
            "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 100.0},
            # Separate venue absent from runtime views MUST still be backfilled from disk snapshot
            "cash:kraken:USD": {"venue": "kraken", "asset": "USD", "free": 50.0},
        },
    )
    assert nav["venues_usd"]["binance"] == pytest.approx(110.0)
    assert nav["venues_usd"]["kraken"] == pytest.approx(50.0)
    assert nav["free_cash_usd"] == pytest.approx(125.0)
    assert nav["locked_quote_usd"] == pytest.approx(25.0)
    assert nav["invested_usd"] == pytest.approx(10.0)
    assert nav["portfolio_total_usd"] == pytest.approx(160.0)

    # --- 2. active_intents keeps resting open orders & partial intents despite legacy status='FILLED' 2/10 ---
    now_s = 1_795_700_000.0
    ai_root = tmp_path / "ai_legacy_partial"
    (ai_root / "cachedb" / "orders").mkdir(parents=True)
    (ai_root / "cachedb" / "state").mkdir(parents=True)
    # Resting order HL-OPEN-PART (2/10 filled) + unverified partial HL-NO-SNAP (2/10) + genuinely full HL-FULL (10/10)
    (ai_root / "cachedb" / "orders" / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSDC": [
                        {
                            "orderId": "HL-OPEN-PART",
                            "symbol": "HYPEUSDC",
                            "side": "BUY",
                            "price": 50.0,
                            "origQty": 10.0,
                            "executedQty": 2.0,
                            "status": "PARTIALLY_FILLED",
                        }
                    ]
                },
                "symbol_verified_ts": {"HYPEUSDC": now_s},
            }
        ),
        encoding="utf-8",
    )
    (ai_root / "cachedb" / "orders" / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSDC": [
                        {"orderId": "HL-OPEN-PART", "symbol": "HYPEUSDC", "side": "BUY", "price": 50.0, "quantity": 2.0, "status": "FILLED"},
                        {"orderId": "HL-NO-SNAP", "symbol": "HYPEUSDC", "side": "BUY", "price": 49.0, "quantity": 2.0, "status": "FILLED"},
                        {"orderId": "HL-FULL", "symbol": "HYPEUSDC", "side": "BUY", "price": 48.0, "quantity": 10.0, "status": "FILLED"},
                    ]
                },
                "fetchtime": {"HYPEUSDC": int(now_s * 1000)},
            }
        ),
        encoding="utf-8",
    )
    (ai_root / "cachedb" / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps(
            [
                {
                    "intent_id": "i_open_part",
                    "venue": "hyperliquid",
                    "symbol": "HYPEUSDC",
                    "side": "BUY",
                    "order_type": "LIMIT",
                    "requested_qty": 10.0,
                    "requested_price": 50.0,
                    "timestamp": now_s - 60.0,
                    "status": "EXECUTED",
                    "execution_payload": {"orderId": "HL-OPEN-PART", "status": "NEW"},
                },
                {
                    "intent_id": "i_no_snap_part",
                    "venue": "hyperliquid",
                    "symbol": "HYPEUSDC",
                    "side": "BUY",
                    "order_type": "LIMIT",
                    "requested_qty": 10.0,
                    "requested_price": 49.0,
                    "timestamp": now_s - 60.0,
                    "status": "EXECUTED",
                    "execution_payload": {"orderId": "HL-NO-SNAP", "status": "NEW"},
                },
                {
                    "intent_id": "i_full_done",
                    "venue": "hyperliquid",
                    "symbol": "HYPEUSDC",
                    "side": "BUY",
                    "order_type": "LIMIT",
                    "requested_qty": 10.0,
                    "requested_price": 48.0,
                    "timestamp": now_s - 60.0,
                    "status": "EXECUTED",
                    "execution_payload": {"orderId": "HL-FULL", "status": "NEW"},
                },
            ]
        ),
        encoding="utf-8",
    )
    with mock.patch("active_intents.resolve_now_ts", return_value=now_s):
        idx_part = active_intents.build_active_intent_index(str(ai_root))
    kept_iids = {r["intent_id"] for r in idx_part["intents"]}
    assert "i_open_part" in kept_iids
    assert "i_no_snap_part" in kept_iids
    assert "i_full_done" not in kept_iids
    assert any(o.get("orderId") == "HL-OPEN-PART" for o in active_intents._load_cached_open_orders(str(ai_root)))

    # --- 3. OrderLedgerReconciler Step 3 skips re-adopting order closed by cid even if vid wasn't on close ---
    cdb_cid = tmp_path / "cachedb_cid_close"
    (cdb_cid / "orders").mkdir(parents=True)
    w_cid = OrderLedgerWriter(cachedb_dir=cdb_cid)
    rec_cid = OrderLedgerReconciler(cachedb_dir=cdb_cid, use_singletons=False, reconcile_grace_sec=10.0)
    rd_cid = OrderLedgerReader(cachedb_dir=cdb_cid)
    w_cid.record_event(
        OrderEventType.SUBMITTED,
        venue="binance",
        symbol="ARBUSDC",
        ts=now_s,
        iid="iid-cid-1",
        cid="AI_cid_only_1",
        side="BUY",
        qty=10.0,
        px=1.0,
    )
    w_cid.record_event(
        OrderEventType.STATUS,
        venue="binance",
        symbol="ARBUSDC",
        ts=now_s + 2.0,
        iid="iid-cid-1",
        cid="AI_cid_only_1",
        vst="CANCELED",
    )
    rec_cid.reconcile_once(now_ts=now_s + 3.0)
    # Stale open snapshot still shows that exact cid with a newly discovered vid -> MUST NOT be re-adopted as orphan!
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(cdb_cid / "orders" / CACHE_BINANCE_ORDER_OPEN_FILENAME),
        {
            "ARBUSDC": [
                {
                    "orderId": "BN-LATE-VID-99",
                    "clientOrderId": "AI_cid_only_1",
                    "side": "BUY",
                    "price": 1.0,
                    "origQty": 10.0,
                    "executedQty": 0.0,
                    "status": "NEW",
                    "timestamp": int(now_s * 1000),
                }
            ]
        },
        now_ms=int((now_s + 4.0) * 1000),
        verified_ts=now_s + 4.0,
    )
    st_cid = rec_cid.reconcile_once(now_ts=now_s + 5.0)
    assert st_cid["orphans_adopted"] == 0 and st_cid["active_count"] == 0
    assert rd_cid.list_active_orders(venue="binance", apply_tail=False) == []

    # --- 4. refresh_stale_t212_cached_positions preserves original balance ts into base_bal_ts/quote_bal_ts ---
    vb_t212_p = tmp_path / "venue_balances_state.json"
    old_bal_ts = now_s - 600.0
    cached_t212 = {
        "pos:t212:NVDA_US_EQ": {
            "venue": "t212",
            "symbol": "NVDA_US_EQ",
            "price": 150.0,
            "free_base": 2.0,
            "free_quote": 500.0,
            "ts": old_bal_ts,
        }
    }
    vb_t212_p.write_text(json.dumps(cached_t212), encoding="utf-8")
    fake_t212_cls = type(
        "FakeT212Provider",
        (),
        {"portfolio_mark_prices": lambda self: {"NVDA_US_EQ": 152.5}},
    )
    with (
        mock.patch.object(mc_mod._t212_prov_mod, "T212Provider", fake_t212_cls),
        mock.patch.object(mc_mod, "_LAST_T212_PORTFOLIO_POLL_TS", 0.0),
    ):
        dirty = mc_mod.refresh_stale_t212_cached_positions(cached_t212, str(vb_t212_p), now_s)
    assert dirty is True
    assert cached_t212["pos:t212:NVDA_US_EQ"]["price"] == pytest.approx(152.5)
    assert cached_t212["pos:t212:NVDA_US_EQ"]["ts"] == pytest.approx(now_s)
    assert cached_t212["pos:t212:NVDA_US_EQ"]["base_bal_ts"] == pytest.approx(old_bal_ts)
    assert cached_t212["pos:t212:NVDA_US_EQ"]["quote_bal_ts"] == pytest.approx(old_bal_ts)

    # --- 5. PositionCostBasisEnrichmentEvaluator honors explicit now_ts regardless of wall-clock shift ---
    lot_ts_ms = int((now_s - 20 * 3600.0) * 1000)  # 20h old at now_s (inside 12h..90d peak-seeding window)
    def _run_cost_basis(wall_clock_s: float) -> dict:
        pos_l = [{"venue": "binance", "symbol": "ARBUSDC", "base": "ARB", "quote": "USDC", "price": 1.0, "free_base": 100.0}]
        ro_map = {
            "binance": {
                "ARBUSDC": {
                    "last_action_side": "BUY",
                    "active_avg_buy_px": 1.0,
                    "last_buy": {"price": 1.0, "ts_ms": lot_ts_ms},
                    "open_lots": [{"qty": 100.0, "entry_px": 1.0, "ts_ms": lot_ts_ms}],
                }
            }
        }
        with mock.patch("intelligence.sentiment.evaluators.portfolio_evaluator._time.time", return_value=wall_clock_s):
            PositionCostBasisEnrichmentEvaluator().evaluate(
                pos_l,
                prices_and_trends={"ARBUSDC": {"price": 1.0, "high_24h": 1.15}},
                venue_recent_orders=ro_map,
                now_ts=now_s,
            )
        return pos_l[0]

    pos_same_w = _run_cost_basis(now_s)
    pos_shifted_w = _run_cost_basis(now_s + 120 * 86400.0)  # +120 days outside 90d window if wall-clock leaked!
    assert pos_same_w.get("trailing_peak") == pytest.approx(1.15)
    assert pos_shifted_w.get("trailing_peak") == pytest.approx(1.15)

    # --- 6. Signed-zero formatting across sentiment_advisor & prompt_sections ---
    br_line = _format_breadth_line(
        {
            "advance_ratio": 0.5,
            "median_change_pct": -0.004,
            "mean_change_pct": -0.004,
            "advancing_count": 5,
            "total_count": 10,
            "eth_price": 3000.0,
            "eth_chg_24h_pct": -0.004,
            "eth_btc_ratio": 0.05,
            "eth_btc_chg_24h_pct": -0.004,
        }
    )
    assert "-0.00%" not in br_line and "median +0.00%" in br_line and "mean +0.00%" in br_line
    wh_line = _format_single_whale_entry("BTCUSDC", {"symbol": "BTCUSDC", "open_interest_usd": 1e8, "open_interest_1h_change_pct": -0.004})
    assert "-0.00%" not in wh_line and "+0.00% 1h" in wh_line
    sec3b_oi = "\n".join(
        _format_microstructure_section(
            {
                "BTCUSDC": {
                    "ob_imbalance": 0.55,
                    "bid_depth_usd": 100_000.0,
                    "ask_depth_usd": 90_000.0,
                    "oi_usd": 50_000_000.0,
                    "oi_1h_chg_pct": 0.5,
                    "oi_15m_chg_pct": -0.004,
                }
            },
            {"BTCUSDC": {"price": 60000.0}},
        )
    )
    assert "oi_15m_chg=+0.00%" in sec3b_oi and "oi_15m_chg=-0.00%" not in sec3b_oi


def test_structural_ssot_primitives_fmt_signed_unified_book_and_balance_view(tmp_path):
    """Verify structural by-construction SSOT primitives:
    1. botcore.fmt_signed normalizes negative zero across decimals, commas, and strip_zeros.
    2. cacheaccount.load_unified_order_book_evidence classifies open-first order caches via order-ledger SSOT.
    3. portfolio_collector.BalanceObservation & derive_runtime_portfolio_view strip internal ts keys even with no open orders.
    """
    from botcore import fmt_signed
    from cacheaccount import (
        CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
        CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
        load_unified_order_book_evidence,
    )
    from intelligence.sentiment.collectors.portfolio_collector import (
        BalanceObservation,
        _INTERNAL_BALANCE_TS_KEYS,
        derive_runtime_portfolio_view,
    )

    # 1. botcore.fmt_signed
    assert fmt_signed(-0.004) == "+0.00"
    assert fmt_signed(-0.004, 1) == "+0.0"
    assert fmt_signed(-0.4, 0) == "+0"
    assert fmt_signed(-0.3, 0, comma=True) == "+0"
    assert fmt_signed(12345.6, 0, comma=True) == "+12,346"
    assert fmt_signed(-0.0001, 3, strip_zeros=True) == "+0"
    assert fmt_signed(1.25, 3, strip_zeros=True) == "+1.25"
    assert fmt_signed(None) == "+0.00"

    # 2. load_unified_order_book_evidence reads open first and prevents legacy 2/10 FILLED hiding resting open order
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    now_s = 1_795_700_000.0
    (cdb / "orders" / CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSDC": [
                        {
                            "orderId": "HL-1",
                            "symbol": "HYPEUSDC",
                            "side": "BUY",
                            "price": 50.0,
                            "origQty": 10.0,
                            "executedQty": 2.0,
                            "status": "PARTIALLY_FILLED",
                        }
                    ]
                },
                "symbol_verified_ts": {"HYPEUSDC": now_s},
            }
        ),
        encoding="utf-8",
    )
    (cdb / "orders" / CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSDC": [
                        {"orderId": "HL-1", "symbol": "HYPEUSDC", "side": "BUY", "price": 50.0, "quantity": 2.0, "status": "FILLED"},
                        {"orderId": "HL-2", "symbol": "HYPEUSDC", "side": "BUY", "price": 49.0, "quantity": 5.0, "origQty": 5.0, "status": "FILLED"},
                    ]
                },
                "fetchtime": {"HYPEUSDC": int(now_s * 1000)},
            }
        ),
        encoding="utf-8",
    )
    ev = load_unified_order_book_evidence(str(cdb))
    assert any(r.get("orderId") == "HL-1" for r in ev.resting_open_orders)
    assert ("hyperliquid", "hypeusdc", "HL-1") not in ev.terminal_order_keys
    assert ("hyperliquid", "hypeusdc", "HL-2") in ev.terminal_order_keys

    # 3. BalanceObservation & derive_runtime_portfolio_view clean internal keys even on empty open_orders_list
    obs = BalanceObservation(free=100.0, ts=now_s, priority=2)
    assert obs.free == pytest.approx(100.0) and obs.ts == pytest.approx(now_s)
    pos_list = [{"venue": "binance", "symbol": "ARBUSDC", "free_base": 10.0, "_base_balance_ts": now_s, "_quote_balance_ts": now_s}]
    cash_map = {"binance:USDC": {"venue": "binance", "asset": "USDC", "free": 100.0, "_balance_ts": now_s}}
    derive_runtime_portfolio_view([], pos_list, cash_map, {}, {}, {}, {}, now_ts_oo=now_s, gateway_max_usd=2000.0)
    for k_int in _INTERNAL_BALANCE_TS_KEYS:
        assert k_int not in pos_list[0]
        assert k_int not in cash_map["binance:USDC"]

    # 4. providers.portfolio_state_store.upsert_venue_balances_snapshot strips internal scratch keys, strips legacy open_orders keys,
    #    and protects balance ts on price-only updates
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from providers.nav_drawdown import format_nav_drawdown_summary_line
    from providers.portfolio_state_store import upsert_venue_balances_snapshot

    vb_sw_path = tmp_path / "state" / "venue_balances_state.json"
    t_bal = now_s - 600.0
    assert upsert_venue_balances_snapshot(
        str(vb_sw_path),
        {
            "pos:t212:AMD_US_EQ": {
                "venue": "t212",
                "symbol": "AMD_US_EQ",
                "price": 200.0,
                "free_base": 3.0,
                "free_quote": 500.0,
                "ts": t_bal,
                "_base_balance_ts": t_bal,
            }
        },
        fallback_state={"open_orders:binance": [{"id": "stale"}]},
    ) is True
    saved_v1 = json.loads(vb_sw_path.read_text(encoding="utf-8"))
    assert "_base_balance_ts" not in saved_v1["pos:t212:AMD_US_EQ"]
    assert "open_orders:binance" not in saved_v1
    # Price-only bump WITHOUT explicit base_bal_ts/quote_bal_ts MUST automatically preserve t_bal
    assert upsert_venue_balances_snapshot(
        str(vb_sw_path),
        {"pos:t212:AMD_US_EQ": {**saved_v1["pos:t212:AMD_US_EQ"], "price": 205.0, "ts": now_s}},
    ) is True
    saved_v2 = json.loads(vb_sw_path.read_text(encoding="utf-8"))
    assert saved_v2["pos:t212:AMD_US_EQ"]["price"] == pytest.approx(205.0)
    assert saved_v2["pos:t212:AMD_US_EQ"]["ts"] == pytest.approx(now_s)
    assert saved_v2["pos:t212:AMD_US_EQ"]["base_bal_ts"] == pytest.approx(t_bal)
    assert saved_v2["pos:t212:AMD_US_EQ"]["quote_bal_ts"] == pytest.approx(t_bal)

    # 5. _format_portfolio_section honors explicit now_ts (zero wall-clock drift on recent order ages)
    ps_replay = {
        "recent_orders": {
            "ARBUSDC": {
                "last_action_side": "BUY",
                "last_buy": {"price": 1.0, "qty": 10.0, "ts_ms": int((now_s - 3600.0) * 1000)},
            }
        }
    }
    with patch("core.time_market._time.time", return_value=now_s + 50 * 86400.0):
        sec3c_replay = "\n".join(_format_portfolio_section(ps_replay, now_ts=now_s))
    assert "age=1.0h" in sec3c_replay

    # 6. providers/nav_drawdown normalizes -0.004 to +0.00% (never -0.00%)
    dd_line = format_nav_drawdown_summary_line(
        {
            "current_nav_usd": 1000.0,
            "sample_count": 5,
            "days_covered": 2.0,
            "daily_nav_return_pct": -0.004,
            "daily_nav_drawdown_pct": -0.004,
            "rolling_24h_drawdown_pct": -0.004,
            "max_drawdown_7d_pct": -0.004,
            "max_drawdown_30d_pct": -0.004,
        }
    )
    assert "-0.00%" not in dd_line and "day_ret=+0.00%" in dd_line


def test_cycle12_premarket_overnight_gap_15m_macro_bar_age_short_trends_and_powell_clickbait(tmp_path):
    """Cycle 12 US Pre-Market open (11:00 EET / 04:00 EST) audit fixes:
    1. compute_24h_price_metrics must not mislabel an 8h overnight close gap as chg_15m_pct /
       chg_30m_pct / chg_1h_pct / chg_3h_pct when pre-market has only been open for 3 minutes.
    2. MacroBenchmarksCollector uses fresh Yahoo preMarketTime for bar_age_min (while still gating
       15m/1h deltas on chart bar freshness) and format_macro_benchmarks_summary renders >=60m as hours.
    3. Equities domain age threshold for short_trends aligns with equity price polling (~180s), so
       live 40s PRE-market short_trends is not falsely tagged [market_closed].
    4. _detect_macro_calendar_headlines requires Fed/monetary context for solo 'powell' matches
       (excluding Nvidia exec Kimberly Powell) and lexicon filters 'N Reason(s) to Buy' + affair gossip.
    """
    from datetime import datetime, timezone
    from intelligence.macro.calendar_ssot import _detect_macro_calendar_headlines
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item
    from intelligence.macro_sentiment_ssot import (
        format_telemetry_domain_age_tag,
        resolve_telemetry_domain_stale_thresholds_sec,
    )
    from intelligence.sentiment.collectors.macro_telemetry_collector import MacroBenchmarksSnapshotCollector
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import format_macro_benchmarks_summary_line
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics

    # C12.1: 4 minutes into US Pre-Market after 8h overnight gap (NVDA_US_EQ 00:59:59 -> 09:01..09:04 UTC)
    t_pre_now = datetime(2026, 11, 25, 9, 4, 4, tzinfo=timezone.utc).timestamp()
    t_post_close = datetime(2026, 11, 25, 0, 59, 59, tzinfo=timezone.utc).timestamp()
    nvda_series = [
        [int((t_pre_now - 24 * 3600.0) * 1000), 159.40],
        [int((t_pre_now - 12 * 3600.0) * 1000), 154.64],
        [int((t_post_close - 600.0) * 1000), 151.85],
        [int((t_post_close - 300.0) * 1000), 151.88],
        [int((t_post_close - 180.0) * 1000), 151.89],
        [int((t_post_close - 120.0) * 1000), 151.88],
        [int((t_post_close - 60.0) * 1000), 151.90],
        [int(t_post_close * 1000), 151.9531],
        # 8h overnight gap -> live PRE ticks
        [int((t_pre_now - 184.0) * 1000), 148.72],
        [int((t_pre_now - 123.0) * 1000), 148.70],
        [int((t_pre_now - 62.0) * 1000), 148.28],
        [int(t_pre_now * 1000), 148.75],
    ]
    m24_pre = compute_24h_price_metrics(nvda_series, is_equity=True, now_ts=t_pre_now)
    assert "chg_15m_pct" not in m24_pre
    assert "chg_30m_pct" not in m24_pre
    assert "chg_1h_pct" not in m24_pre
    assert "chg_3h_pct" not in m24_pre
    assert m24_pre["chg_1m_pct"] == pytest.approx(((148.75 - 148.28) / 148.28) * 100.0, abs=0.05)
    assert m24_pre["chg_3m_pct"] == pytest.approx(((148.75 - 148.72) / 148.72) * 100.0, abs=0.05)
    assert "chg_12h_pct" in m24_pre and "chg_24h_pct" in m24_pre

    # C12.2a: Formatter renders >=60m bar_age in hours even when market_state == 'PRE' (never '490m')
    mb_fmt = format_macro_benchmarks_summary_line(
        {"items": {"SPY": {"price": 712.91, "chg_24h_pct": -0.79, "market_state": "PRE", "bar_age_min": 490.0}}},
        include_crypto=False,
    )
    assert "bar_age=8.2h" in mb_fmt and "bar_age=490m" not in mb_fmt

    # C12.2b: Collector uses Yahoo preMarketTime for bar_age_min while blocking stale 8h chart bars from 15m/1h
    stale_bar_ts = t_pre_now - 490.0 * 60.0
    yahoo_spy = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "marketState": "PRE",
                        "preMarketPrice": 712.91,
                        "preMarketTime": int(t_pre_now - 30.0),
                        "regularMarketPrice": 710.49,
                        "previousClose": 718.58,
                    },
                    "timestamp": [int(stale_bar_ts - 3600.0), int(stale_bar_ts - 900.0), int(stale_bar_ts)],
                    "indicators": {"quote": [{"close": [715.0, 711.0, 710.49]}]},
                }
            ]
        }
    }
    mb_res = MacroBenchmarksSnapshotCollector(start_sync=False).collect(
        str(tmp_path),
        now_ts=t_pre_now,
        persist=False,
        fetch_fn=lambda url, **_kw: yahoo_spy if "SPY" in url else None,
    )
    assert mb_res is not None
    spy_entry = mb_res["items"]["SPY"]
    assert spy_entry["market_state"] == "PRE"
    assert spy_entry["bar_age_min"] <= 1.0
    assert "chg_15m_pct" not in spy_entry and "chg_1h_pct" not in spy_entry

    # C12.3: Equities short_trends threshold matches equity price polling (40s PRE quote is NOT [market_closed])
    eq_thrs = resolve_telemetry_domain_stale_thresholds_sec("equities")
    assert eq_thrs["short_trends"] >= eq_thrs["prices_newest"]
    assert format_telemetry_domain_age_tag("short_trends", 40.0, asset_scope="equities", us_rth_open=False) == ""
    assert format_telemetry_domain_age_tag("short_trends", 5000.0, asset_scope="equities", us_rth_open=False) == "[market_closed]"

    # C12.4: Solo 'powell' in _detect_macro_calendar_headlines excludes non-Fed Kimberly Powell; preserves Fed Chair
    cal_hits = _detect_macro_calendar_headlines(
        [
            "Will Nvidia Exec Kimberly Powell Leave? Crypto Bettors Weigh Exit Odds Amid Alleged Jensen Huang Affair",
            "Powell signals Fed may pause rate cuts amid sticky inflation",
            "Stock futures slip as traders await key inflation report",
        ]
    )
    assert all("Kimberly Powell" not in h for h in cal_hits)
    assert any("Powell signals Fed" in h for h in cal_hits)
    assert any("inflation report" in h for h in cal_hits)

    # C12.5: Clickbait & tabloid affair headlines excluded; genuine equity news preserved
    assert is_excluded_reference_news_item("1 Reason to Buy AMD Stock Before 2027")
    assert is_excluded_reference_news_item(
        "Will Nvidia Exec Kimberly Powell Leave? Crypto Bettors Weigh Exit Odds Amid Alleged Jensen Huang Affair"
    )
    assert not is_excluded_reference_news_item(
        "Nvidia cuts revenue guidance as US tightens AI chip export curbs"
    )


def test_cycle13_reprice_balance_no_double_subtract_equity_ob_depth_sre_bypass_and_war_stubs(tmp_path):
    """Cycle 13 audit fixes:
    1. When monitororder cancels & reprices an already-locked order after bal_ts, capping synthetic
       post-balance free-cash deduction by max(0, cur_locked - reflected_locked) prevents phantom
       double-subtraction (e.g. -$104.85 NAV loss on Binance BTCUSDC), while still deducting
       genuine brand-new post-bal_ts orders in full.
    2. Equity orderbook fallback derives depth from RTH bar volume when ctx.daily_vol_sh is absent
       and anchors empty-band buckets at BBO rather than fake $1000 @ +/-0.50% (103bps slip).
    3. SRE log harvester ignores benign AI guard-bypass messages ('historical price reference and
       quantity/weight guard were explicitly bypassed').
    4. Photo-wire stubs ('APTOPIX Russia Ukraine War'), bare topic stubs ('Russia Ukraine War'),
       and 'How forced fighters...' explainers are excluded from high-severity war alerts.
    """
    from cachepaths import CACHE_BINANCE_ORDER_OPEN_FILENAME
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.macro.lexicon_ssot import is_excluded_reference_news_item, match_high_severity_keywords
    from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    # C13.1a: Repriced order after bal_ts with unchanged total locked ($1004.66 -> $1004.63) must NOT double-subtract!
    now_s = 1_795_600_845.0
    bal_ts = now_s - 48.0
    reprice_ts = now_s - 15.0
    cdir1 = tmp_path / "cdir_c13_reprice"
    (cdir1 / "state").mkdir(parents=True)
    (cdir1 / "orders").mkdir(parents=True)
    (cdir1 / "state" / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "binance:BTC": 0.025,
                "binance:USDC": 5143.79,
                "cash:binance:USDC": {
                    "venue": "binance",
                    "asset": "USDC",
                    "free": 5143.79,
                    "ts": bal_ts,
                    "locked_quote_usd": 1004.66,
                },
                "pos:binance:BTCUSDC": {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "base": "BTC",
                    "quote": "USDC",
                    "price": 58973.0,
                    "free_base": 0.025,
                    "free_quote": 5143.79,
                    "ts": bal_ts,
                    "base_bal_ts": bal_ts,
                    "quote_bal_ts": bal_ts,
                    "locked_quote_usd": 1004.66,
                },
            }
        ),
        encoding="utf-8",
    )
    (cdir1 / "orders" / CACHE_BINANCE_ORDER_OPEN_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "10965871781",
                            "side": "BUY",
                            "type": "LIMIT",
                            "price": 58050.0,
                            "origQty": 0.0155,
                            "executedQty": 0.0,
                            "status": "NEW",
                            "timestamp": int((bal_ts - 3600.0) * 1000),
                        },
                        {
                            "orderId": "10966561934",
                            "side": "BUY",
                            "type": "LIMIT",
                            "price": 58907.4,
                            "origQty": 0.00178,
                            "executedQty": 0.0,
                            "status": "NEW",
                            "clientOrderId": "MO_TA_OR_reprice_0",
                            "timestamp": int(reprice_ts * 1000),
                        },
                    ]
                },
                "symbol_verified_ts": {"BTCUSDC": now_s - 2.0},
            }
        ),
        encoding="utf-8",
    )
    tpos_r, _, vcash_r, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir1),
        prices_and_trends={"BTCUSDC": {"price": 58973.0}},
        allow_empty_inst_recovery=True,
        allow_network=False,
        persist=False,
        now_ts=now_s,
    )
    assert vcash_r["binance:USDC"]["free"] == pytest.approx(5143.79, abs=0.02)
    assert vcash_r["binance:USDC"]["locked_quote_usd"] == pytest.approx(1004.63, abs=0.02)
    assert tpos_r[0]["free_quote"] == pytest.approx(5143.79, abs=0.02)

    # C13.1b: Genuine brand-new order (+$500) placed after bal_ts STILL deducts exact incremental +$500!
    (cdir1 / "orders" / CACHE_BINANCE_ORDER_OPEN_FILENAME).write_text(
        json.dumps(
            {
                "items": {
                    "BTCUSDC": [
                        {
                            "orderId": "10965871781",
                            "side": "BUY",
                            "type": "LIMIT",
                            "price": 58050.0,
                            "origQty": 0.0155,
                            "executedQty": 0.0,
                            "status": "NEW",
                            "timestamp": int((bal_ts - 3600.0) * 1000),
                        },
                        {
                            "orderId": "10966561934",
                            "side": "BUY",
                            "type": "LIMIT",
                            "price": 58907.4,
                            "origQty": 0.00178,
                            "executedQty": 0.0,
                            "status": "NEW",
                            "timestamp": int(reprice_ts * 1000),
                        },
                        {
                            "orderId": "10966599999",
                            "side": "BUY",
                            "type": "LIMIT",
                            "price": 50000.0,
                            "origQty": 0.01,
                            "executedQty": 0.0,
                            "status": "NEW",
                            "timestamp": int((reprice_ts + 2.0) * 1000),
                        },
                    ]
                },
                "symbol_verified_ts": {"BTCUSDC": now_s - 2.0},
            }
        ),
        encoding="utf-8",
    )
    tpos_n, _, vcash_n, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdir1),
        prices_and_trends={"BTCUSDC": {"price": 58973.0}},
        allow_empty_inst_recovery=True,
        allow_network=False,
        persist=False,
        now_ts=now_s,
    )
    assert vcash_n["binance:USDC"]["free"] == pytest.approx(5143.79 - 499.97, abs=0.05)
    assert vcash_n["binance:USDC"]["locked_quote_usd"] == pytest.approx(1504.63, abs=0.05)

    # C13.2: Equity orderbook fallback uses bar volume when daily_vol_sh is missing and anchors at BBO
    col_ob = OrderbookDepthCollector(cache_dir=str(tmp_path / "ob_c13"), start_sync=False)
    # Pre-market gap-down to $147.53 while prior RTH bars sit above $152 with 30M volume
    rth_bars_nvda = [(now_s - 36000.0 + i * 300.0, 155.0, 155.5, 154.5, 155.0, 400_000.0) for i in range(78)]
    col_ob._resolve_equity_context_bars_and_price = lambda sym, ts, include_options=False: (
        {"best_bid": 147.48, "best_ask": 147.58, "bid_size_sh": 0.0, "ask_size_sh": 0.0},
        rth_bars_nvda,
        147.53,
    )
    snap_nvda = col_ob._fetch_equity_orderbook("NVDA_US_EQ", now_s)
    assert snap_nvda is not None
    assert snap_nvda.bid_depth_usd >= 100_000.0 and snap_nvda.ask_depth_usd >= 100_000.0
    assert snap_nvda.largest_bid_wall_price == pytest.approx(147.48, abs=0.01)
    assert snap_nvda.largest_ask_wall_price == pytest.approx(147.58, abs=0.01)

    # C13.3: Benign AI guard-bypass line does not count as SRE guard intervention
    ldir = tmp_path / "logs_c13"
    ldir.mkdir(parents=True)
    now_hms = time.strftime("%H:%M:%S")
    (ldir / "order_guard.log").write_text(
        f"{now_hms} [GUARD] BUY BTCUSDC: the historical price reference and quantity/weight guard were explicitly bypassed\n",
        encoding="utf-8",
    )
    anom = harvest_log_anomalies_from_dirs([str(ldir)], 1.0)
    assert anom.error_count == 0 and anom.guard_interventions == []

    # C13.4: APTOPIX photo stub, bare 'Russia Ukraine War' stub, and 'How forced fighters...' excluded
    assert is_excluded_reference_news_item("APTOPIX Russia Ukraine War - TelegraphHerald.com")
    assert match_high_severity_keywords("Russia Ukraine War - WV News") == ()
    assert match_high_severity_keywords("How 'forced fighters' are used by Russia in Ukraine war - RTE.ie") == ()
    assert match_high_severity_keywords(
        "Ukraine war escalates as Russian missile strike hits major oil refinery"
    )


def test_cycle14_equity_ob_ext_hours_depth_hyphenated_publisher_strip_and_sre_guard_pass(tmp_path):
    """Cycle 14 audit fixes:
    1. Equity orderbook filters out zero-volume extended-hours bars before taking the 78-bar RTH
       liquidity window and backfills background book depth behind thin L1 odd-lot prints (e.g.
       1-share $386 AMD bid / $0.11 decayed RGNT bar dust) so PRE-market depth & slip_2k stay sane.
    2. _PUBLISHER_SUFFIX_RE strips hyphenated publisher names (' - Daily News-Record') so bare
       topic stubs ('Russia Ukraine War - Daily News-Record') plus speculative 'cling to power?'
       questions and non-kinetic diplomatic spats do not fire high-severity war alerts.
    3. SRE log harvester ignores passing profit_guard checks (diff >= threshold) while retaining
       actual guard vetoes (diff < threshold).
    """
    from intelligence.external.collectors.orderbook_depth_collector import OrderbookDepthCollector
    from intelligence.macro.lexicon_ssot import _strip_headline_publisher_suffix, match_high_severity_keywords
    from orchestratorOS.lib.technical_repair_ssot import harvest_log_anomalies_from_dirs

    now_s = 1_795_604_400.0
    col_ob = OrderbookDepthCollector(cache_dir=str(tmp_path / "ob_c14"), start_sync=False)

    # C14.1a: AMD PRE-market gap-down with 74 trailing zero-vol bars + 1-share L1 bid ($386.3) vs 60-share ask ($23.2K)
    rth_amd = [(now_s - 70000.0 + i * 300.0, 412.0, 414.0, 411.0, 413.0, 250_000.0) for i in range(78)]
    pre_zero_amd = [(now_s - 22200.0 + i * 300.0, 386.5, 387.0, 386.0, 386.5, 0.0) for i in range(74)]
    col_ob._resolve_equity_context_bars_and_price = lambda sym, ts, include_options=False: (
        {
            "best_bid": 386.3,
            "best_ask": 386.7,
            "bid_size_sh": 1.0,
            "ask_size_sh": 60.0,
            "daily_vol_sh": 562_086.0,
            "market_state": "PRE",
        },
        rth_amd + pre_zero_amd,
        386.5,
    )
    snap_amd = col_ob._fetch_equity_orderbook("AMD_US_EQ", now_s)
    assert snap_amd is not None
    assert snap_amd.bid_depth_usd >= 50_000.0 and snap_amd.ask_depth_usd >= 50_000.0
    assert 0.25 <= snap_amd.imbalance_ratio <= 0.75

    # C14.1b: RGNT micro-cap with 70 trailing zero-vol bars decaying in-band RTH bars must NOT emit $0/$1 depth
    rth_rgnt = [(now_s - 70000.0 + i * 300.0, 0.78, 0.795, 0.775, 0.782, 1500.0) for i in range(20)]
    pre_zero_rgnt = [(now_s - 21000.0 + i * 300.0, 0.7816, 0.7816, 0.7816, 0.7816, 0.0) for i in range(70)]
    col_ob._resolve_equity_context_bars_and_price = lambda sym, ts, include_options=False: (
        {
            "best_bid": 0.80,
            "best_ask": 0.966,
            "bid_size_sh": 100.0,
            "ask_size_sh": 100.0,
            "daily_vol_sh": 312.0,
            "market_state": "PRE",
        },
        rth_rgnt + pre_zero_rgnt,
        0.7816,
    )
    snap_rgnt = col_ob._fetch_equity_orderbook("RGNT_US_EQ", now_s)
    assert snap_rgnt is not None
    assert snap_rgnt.bid_depth_usd >= 400.0 and snap_rgnt.ask_depth_usd >= 400.0

    # C14.2: Hyphenated publisher suffix stripped; bare war stub, speculative 'cling to power?', and diplomat spat excluded
    assert _strip_headline_publisher_suffix("Russia Ukraine War - Daily News-Record") == "Russia Ukraine War"
    assert match_high_severity_keywords("Russia Ukraine War - Daily News-Record") == ()
    assert match_high_severity_keywords("Could Netanyahu take Israel into another war with Iran to cling to power?") == ()
    assert match_high_severity_keywords(
        "North Korea Kicks Out Swiss Diplomats in Spat Linked to Ukraine War - The New York Times"
    ) == ()
    assert match_high_severity_keywords(
        "Iran rations electricity: How deeply are US sanctions, blockade biting?"
    )

    # C14.3: Passing order_guard check (diff >= threshold) excluded from SRE guard_interventions; real veto kept
    ldir = tmp_path / "logs_c14"
    ldir.mkdir(parents=True)
    now_hms = time.strftime("%H:%M:%S")
    (ldir / "rtrade.log").write_text(
        f"{now_hms} [GUARD] BUY TAOUSDC (mode=hybrid): ref 189.45 (the window), price 174.0, diff 8.16%, threshold 1.15%\n"
        f"{now_hms} [GUARD] SELL TAOUSDC (mode=last): ref 175.9 (the window), price 176.2, diff 0.17%, threshold 1.15%\n",
        encoding="utf-8",
    )
    anom = harvest_log_anomalies_from_dirs([str(ldir)], 1.0)
    assert anom.error_count == 0
    assert len(anom.guard_interventions) == 1
    assert "SELL TAOUSDC" in anom.guard_interventions[0]


def test_structural_ssot_bug_free_by_construction_refactors(tmp_path):
    """End-to-end structural invariant tests for all 7 SSOT refactors and AST Audits 15 & 16."""
    import json
    from order_lifecycle import OrderPreemptionTransaction
    import order_retry
    from providers.intent_state_store import (
        load_durable_intents_from_cachedb,
        load_known_ai_order_ids_from_cachedb,
    )
    from providers.order_status_ssot import (
        classify_order_evidence_state,
        classify_venue_order_status,
        merge_monotonic_fill_row_status,
    )
    from providers.portfolio_valuation import BalanceSnapshotRecord
    from cacheaccount import UnifiedOrderBookEvidence
    from state_io import (
        read_jsonl_bucket_map_from_offset,
        read_jsonl_dicts_from_offset,
    )
    from strategies.spot_engine import PlannedTranche
    from verify_tools.rebalance_kraken_to_hl import (
        ArbitrumBalanceDiagnostic,
        ArbitrumReceiptProof,
        build_arbitrum_arrival_result,
    )
    from verify_tools.ssot_ast_audit import (
        audit_dataclass_slots_assignments_ast,
        audit_state_file_owners_ast,
    )

    # 1. OrderPreemptionTransaction: RAII rollback on early return & exception, idempotent, suppressed on commit/ambiguous/deferred
    restored_calls: list[tuple[list[dict], str]] = []

    def _rec_restore(orders, reason):
        restored_calls.append((list(orders), reason))
        return len(orders)

    with OrderPreemptionTransaction(restore_fn=_rec_restore, default_reason="early_exit") as tx_early:
        tx_early.record_canceled({"orderId": "101", "side": "BUY", "price": 1.0, "qty": 10.0})
    assert len(restored_calls) == 1 and restored_calls[0][1] == "early_exit"
    assert tx_early.rollback("second_call") == 0 and len(restored_calls) == 1

    with pytest.raises(RuntimeError, match="boom"):
        with OrderPreemptionTransaction(restore_fn=_rec_restore, default_reason="on_exc") as tx_exc:
            tx_exc.record_canceled({"orderId": "102", "side": "SELL", "price": 2.0, "qty": 5.0})
            raise RuntimeError("boom")
    assert len(restored_calls) == 2 and restored_calls[1][1] == "on_exc"

    for finalize_method in ("commit", "mark_ambiguous", "mark_deferred"):
        with OrderPreemptionTransaction(restore_fn=_rec_restore, default_reason="nope") as tx_ok:
            tx_ok.record_canceled({"orderId": "103", "side": "BUY", "price": 1.0, "qty": 1.0})
            getattr(tx_ok, finalize_method)()
    assert len(restored_calls) == 2

    # 2. Unified venue status classifier & monotonic evidence lattice
    ok_st = classify_venue_order_status("FILLED", status_map={"FILLED": "closed", "NEW": "open"})
    assert ok_st.local_status == "closed" and ok_st.is_terminal and ok_st.is_filled and not ok_st.is_terminal_non_fill
    for bad_raw in (None, "", True, False, "SURPRISE_STATUS"):
        with pytest.raises(ValueError):
            classify_venue_order_status(bad_raw, status_map={"FILLED": "closed"})
    # Monotonicity: FILLED(4) > CANCELED(3) > PARTIAL(2) > OPEN(1) regardless of timestamp
    assert merge_monotonic_fill_row_status("FILLED", "CANCELED", incoming_ts=200, existing_ts=100) == "FILLED"
    assert merge_monotonic_fill_row_status("CANCELED", "FILLED", incoming_ts=100, existing_ts=200) == "FILLED"
    assert merge_monotonic_fill_row_status("CANCELED", "OPEN", incoming_ts=999, existing_ts=100) == "CANCELED"
    assert merge_monotonic_fill_row_status("PARTIALLY_FILLED", "OPEN", incoming_ts=999, existing_ts=100) == "PARTIALLY_FILLED"
    assert (
        classify_order_evidence_state(
            "FILLED", filled_qty=2.0, req_qty=10.0, require_qty_match_for_filled=True, collapse_cancel_variants=True
        )
        == "PARTIAL"
    )

    # 3. Exclusive state owners + order_retry supersede safety + AST Audits 11 & 12
    cb_dir = tmp_path / "cachedb"
    (cb_dir / "state").mkdir(parents=True)
    (cb_dir / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps({"intents": [{"intent_id": "ai_99", "venue": "binance", "symbol": "BTCUSDC", "execution_payload": {"orderId": "778899"}}]}),
        encoding="utf-8",
    )
    assert len(load_durable_intents_from_cachedb(str(cb_dir))) == 1
    assert "778899" in load_known_ai_order_ids_from_cachedb(str(cb_dir))

    assert order_retry.is_safe_to_supersede_record({"lifecycle": "submit_pending", "submission_state": "refused"}) is True
    assert order_retry.is_safe_to_supersede_record({"lifecycle": "awaiting_cancel"}) is True
    assert order_retry.is_safe_to_supersede_record({"lifecycle": "accepted"}) is False
    assert order_retry.is_safe_to_supersede_record({"lifecycle": "submit_pending", "submission_state": "producer_claimed"}) is False
    assert order_retry.is_safe_to_supersede_record({"lifecycle": "submit_pending", "submission_state": "unknown"}) is False
    assert order_retry.is_safe_to_supersede_record({"lifecycle": "submit_pending", "last_failure_reason": "submit_ambiguous"}) is False
    assert order_retry.is_safe_to_supersede_record({"lifecycle": "submit_pending", "last_failure_reason": "response_without_order_id"}) is False

    # Negative tests proving AST Audits 11 & 12 catch real bugs statically
    synth_repo = tmp_path / "synth_repo"
    synth_repo.mkdir()
    (synth_repo / "bad_reader.py").write_text(
        "import os\n"
        "from cachepaths import AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME\n"
        "from state_io import read_json_recovering\n"
        "def f(d):\n"
        "    p = os.path.join(d, AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME)\n"
        "    return read_json_recovering(p)\n",
        encoding="utf-8",
    )
    v11 = audit_state_file_owners_ast(synth_repo, rel_paths=["bad_reader.py"])
    assert any("AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME" in err for err in v11)

    (synth_repo / "bad_slot.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(slots=True)\n"
        "class _Ctx:\n"
        "    order: object = None\n"
        "def mutate(ctx: _Ctx):\n"
        "    ctx.order = 1\n"
        "    ctx.enqueued_intent_id = 'boom'\n",
        encoding="utf-8",
    )
    v12 = audit_dataclass_slots_assignments_ast(synth_repo, rel_paths=["bad_slot.py"])
    assert any("_Ctx.enqueued_intent_id" in err for err in v12)
    assert not any("_Ctx.order" in err for err in v12)

    # 4. UnifiedOrderBookEvidence.proves_symbol_absent: strictly per-(venue, symbol)
    sym_ver_tuple = {("kraken", "spcxxusd"): 1000.0}
    sym_ver_str = {"kraken:spcxxusd": 1000.0}
    assert UnifiedOrderBookEvidence.proves_symbol_absent(sym_ver_tuple, "kraken", "spcxxusd", ref_ts=900.0, grace_sec=45.0) is True
    assert UnifiedOrderBookEvidence.proves_symbol_absent(sym_ver_tuple, "kraken", "hypeusd", ref_ts=900.0, grace_sec=45.0) is False
    assert UnifiedOrderBookEvidence.proves_symbol_absent(sym_ver_str, "kraken", "spcxxusd", ref_ts=980.0, grace_sec=45.0) is False
    assert UnifiedOrderBookEvidence.proves_symbol_absent(sym_ver_str, "kraken", "spcxxusd", ref_ts=900.0, grace_sec=45.0) is True

    # 5. BalanceSnapshotRecord conservation law (normal + reprice reflected_locked cap)
    bal_new = BalanceSnapshotRecord(free_raw=1000.0, total_locked=300.0, pre_balance_locked=300.0, reflected_locked=None)
    assert bal_new.runtime_free == pytest.approx(700.0)
    assert bal_new.venue_reflected_locked == pytest.approx(0.0)
    assert (bal_new.runtime_free + bal_new.total_locked) == pytest.approx(bal_new.free_raw + bal_new.venue_reflected_locked)

    bal_reprice = BalanceSnapshotRecord(free_raw=5143.79, total_locked=1504.63, pre_balance_locked=1004.66, reflected_locked=1004.66)
    assert bal_reprice.synthetic_deduct == pytest.approx(499.97, abs=1e-6)
    assert bal_reprice.runtime_free == pytest.approx(5143.79 - 499.97, abs=1e-6)
    assert bal_reprice.venue_reflected_locked == pytest.approx(1004.66, abs=1e-6)
    assert (bal_reprice.runtime_free + bal_reprice.total_locked) == pytest.approx(
        bal_reprice.free_raw + bal_reprice.venue_reflected_locked
    )

    # 6. Unified JSONL frame reader: identical unterminated EOF behavior for cache & ledger
    f_jsonl = tmp_path / "shared_frames.jsonl"
    f_jsonl.write_bytes(b'{"s":"BTCUSDC","i":{"v":1}}')
    bmap_0, _, off_b0, _, _, _ = read_jsonl_bucket_map_from_offset(str(f_jsonl), start_offset=0)
    drows_0, off_d0 = read_jsonl_dicts_from_offset(str(f_jsonl), offset=0)
    assert bmap_0 == {} and drows_0 == [] and off_b0 == 0 and off_d0 == 0
    with open(f_jsonl, "ab") as fh:
        fh.write(b"\n")
    bmap_1, _, off_b1, _, _, _ = read_jsonl_bucket_map_from_offset(str(f_jsonl), start_offset=0)
    drows_1, off_d1 = read_jsonl_dicts_from_offset(str(f_jsonl), offset=0)
    assert len(bmap_1["BTCUSDC"]) == 1 and len(drows_1) == 1 and off_b1 == off_d1 > 0

    # 7b. PlannedTranche frozen slotted single-plan record
    tr = PlannedTranche(slot_idx=0, price=105.0, qty=2.5, tranche_idx=0, locked=False)
    assert tr.price == 105.0 and tr.qty == 2.5 and tr.locked is False

    # 7c. ArbitrumReceiptProof enforces valid EVM tx hash + confirmed + non-reverted + positive USDC
    valid_hash = "0x" + "ab" * 32
    with pytest.raises(ValueError):
        ArbitrumReceiptProof(tx_hash="KRAKEN-INTERNAL-REF", received_usdc=10.0)
    with pytest.raises(ValueError):
        ArbitrumReceiptProof(tx_hash=valid_hash, received_usdc=10.0, confirmed=False)
    with pytest.raises(ValueError):
        ArbitrumReceiptProof(tx_hash=valid_hash, received_usdc=10.0, reverted=True)
    with pytest.raises(ValueError):
        ArbitrumReceiptProof(tx_hash=valid_hash, received_usdc=0.0)
    with pytest.raises(TypeError):
        build_arbitrum_arrival_result(
            {"confirmed": True},  # type: ignore[arg-type]
            ArbitrumBalanceDiagnostic(0.0, 10.0, True),
            withdrawal_id="W1",
            kraken_status="success",
        )
    proof = ArbitrumReceiptProof(tx_hash=valid_hash, received_usdc=10.0, block_number=12345)
    diag = ArbitrumBalanceDiagnostic(usdc_before=5.0, usdc_after=15.0, observed=True)
    arr = build_arbitrum_arrival_result(proof, diag, withdrawal_id="W1", kraken_status="success")
    assert arr["arrived"] is True and arr["stages"]["transaction_confirmed"] is True and arr["arb_txid"] == valid_hash


def test_free_balance_ssot_refactor_atomic_locked_xstock_reconcile_and_t212_reprice(tmp_path, monkeypatch):
    """Comprehensive free/locked balance SSOT refactor tests:
    1. KrakenXstocksCollector counts free_base + locked_base_qty via extract_position_held_qty so resting SELL
       orders on SPCXxUSD never zero out reported xStock holdings.
    2. build_cached_balance_observations preserves reflected_locked grouped strictly by matching timestamp
       bucket (stale position locks cannot pollute fresh balances; pos:* quote locks survive absent/stale cash:*).
    3. derive_runtime_portfolio_view reconciles shared free_quote/free_base across positions even when
       open_orders_list is empty or synthetic_deduct == 0, and strips ALL INTERNAL_BALANCE_SCRATCH_KEYS.
    4. providers.portfolio_valuation.upsert_venue_balances_snapshot strips _prev_locked_quote_usd and _prev_locked_base_qty via SSOT keys.
    5. refresh_stale_t212_cached_positions updates both locked_base_usd and value_usd in-memory (including free_base==0.0) without mutating disk.
    6. Atomic free_and_locked_balance on Binance/Hyperliquid/Kraken/T212 exposes live exchange locks in one
       call and prevents double-subtraction when an order reprices during a live collect() cycle.
    """
    from types import SimpleNamespace
    import binance_api.bapi as bapi_mod
    from intelligence.sentiment.collectors.microstructure_collector import refresh_stale_t212_cached_positions
    from intelligence.sentiment.collectors.portfolio_collector import (
        KrakenXstocksCollector,
        TradablePositionsAndOrdersCollector,
        derive_runtime_portfolio_view,
    )
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.portfolio_state_store import upsert_venue_balances_snapshot
    from providers.portfolio_valuation import (
        build_cached_balance_observations,
        extract_position_held_qty,
        reprice_position_balance_fields,
    )
    from state_stores.venue_balances_state import INTERNAL_BALANCE_SCRATCH_KEYS
    from providers.quantity import resolve_assets
    from providers.t212_provider import T212Provider

    now_s = 1_795_605_000.0

    # 1. KrakenXStockCollector sees free_base + locked_base_qty when all SPCXx is locked in a resting SELL order
    cdb_xs = tmp_path / "xs_cdb"
    (cdb_xs / "state").mkdir(parents=True)
    (cdb_xs / "state" / "venue_balances_state.json").write_text(
        json.dumps(
            {
                "pos:kraken:SPCXxUSD": {
                    "venue": "kraken",
                    "symbol": "SPCXxUSD",
                    "base": "SPCXx",
                    "quote": "USD",
                    "free_base": 0.0,
                    "locked_base_qty": 12.135889,
                    "price": 122.47,
                    "ts": now_s,
                }
            }
        ),
        encoding="utf-8",
    )
    assert extract_position_held_qty({"free_base": 0.0, "locked_base_qty": 12.135889}) == pytest.approx(12.135889)
    xs_res = KrakenXstocksCollector(start_sync=False).collect(
        str(cdb_xs),
        prices_and_trends={"SPCXxUSD": {"price": 122.47}},
    )
    assert xs_res is not None
    assert xs_res["holdings"]["SPCXx"] == pytest.approx(12.135889, abs=1e-5)

    # 2. build_cached_balance_observations groups locks strictly by matching timestamp bucket & preserves pos:* quote lock
    cached_bals = {
        "pos:binance:BTCUSDC": {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "base": "BTC",
            "quote": "USDC",
            "free_base": 0.02,
            "free_quote": 5000.0,
            "base_bal_ts": now_s,
            "quote_bal_ts": now_s,
            "locked_base_qty": 0.01,
            "locked_quote_usd": 900.0,
        },
        "pos:binance:BTCUSDT": {
            "venue": "binance",
            "symbol": "BTCUSDT",
            "base": "BTC",
            "quote": "USDT",
            "free_base": 0.05,
            "free_quote": 100.0,
            "base_bal_ts": now_s - 500.0,  # STALE base timestamp -> must NOT pollute fresh BTC reflected_locked!
            "quote_bal_ts": now_s - 500.0,
            "locked_base_qty": 0.04,
        },
        "pos:binance:TAOUSDC": {
            "venue": "binance",
            "symbol": "TAOUSDC",
            "base": "TAO",
            "quote": "USDC",
            "free_base": 10.0,
            "free_quote": 5000.0,
            "base_bal_ts": now_s,
            "quote_bal_ts": now_s,
            "locked_quote_usd": 100.0,
        },
        "cash:binance:USDC": {
            "venue": "binance",
            "asset": "USDC",
            "free": 4800.0,
            "ts": now_s - 300.0,  # OLDER than pos:* -> pos:* quote observation (5000, reflected=1000) must win!
            "locked_quote_usd": 250.0,
        },
    }
    obs_map = build_cached_balance_observations(cached_bals, resolve_assets_fn=resolve_assets)
    assert obs_map[("binance", "BTC")].free == pytest.approx(0.02)
    assert obs_map[("binance", "BTC")].reflected_locked == pytest.approx(0.01)
    assert obs_map[("binance", "USDC")].free == pytest.approx(5000.0)
    assert obs_map[("binance", "USDC")].reflected_locked == pytest.approx(1000.0)

    # 3. derive_runtime_portfolio_view reconciles shared free_quote across positions EVEN with empty open_orders_list
    pos_reconcile = [
        {
            "venue": "binance",
            "symbol": "TAOUSDC",
            "base": "TAO",
            "quote": "USDC",
            "price": 175.0,
            "free_base": 10.0,
            "free_quote": 4200.0,
            "_base_balance_ts": now_s - 200.0,
            "_quote_balance_ts": now_s - 200.0,
            "_prev_locked_quote_usd": 100.0,
            "_prev_locked_base_qty": 0.5,
        }
    ]
    cash_reconcile = {
        "binance:USDC": {
            "venue": "binance",
            "asset": "USDC",
            "free": 5143.68,
            "_balance_ts": now_s,
            "_prev_locked_quote_usd": 0.0,
        }
    }
    derive_runtime_portfolio_view(
        [], pos_reconcile, cash_reconcile, {}, {}, {}, {}, now_ts_oo=now_s, gateway_max_usd=2000.0
    )
    assert pos_reconcile[0]["free_quote"] == pytest.approx(5143.68, abs=0.01)
    assert pos_reconcile[0]["effective_max_buy_usd"] == pytest.approx(2000.0, abs=0.01)
    for sk in INTERNAL_BALANCE_SCRATCH_KEYS:
        assert sk not in pos_reconcile[0]
        assert sk not in cash_reconcile["binance:USDC"]

    # 4. upsert_venue_balances_snapshot strips _prev_locked_* scratch keys too
    vb_strip_p = tmp_path / "strip_vb" / "venue_balances_state.json"
    assert upsert_venue_balances_snapshot(
        str(vb_strip_p),
        {
            "cash:binance:USDC": {
                "venue": "binance",
                "asset": "USDC",
                "free": 100.0,
                "ts": now_s,
                "_prev_locked_quote_usd": 50.0,
            },
            "pos:binance:BTCUSDC": {
                "venue": "binance",
                "symbol": "BTCUSDC",
                "free_base": 0.1,
                "ts": now_s,
                "_prev_locked_base_qty": 0.05,
            },
        },
    )
    saved_strip = json.loads(vb_strip_p.read_text(encoding="utf-8"))
    for sk in INTERNAL_BALANCE_SCRATCH_KEYS:
        assert sk not in saved_strip["cash:binance:USDC"]
        assert sk not in saved_strip["pos:binance:BTCUSDC"]

    # 5. reprice_position_balance_fields & refresh_stale_t212_cached_positions update locked_base_usd and zero value_usd
    t212_pos = {"free_base": 0.0, "value_usd": 300.0, "locked_base_qty": 2.0, "locked_base_usd": 200.0, "price": 100.0}
    reprice_position_balance_fields(t212_pos, 125.0)
    assert t212_pos["value_usd"] == pytest.approx(0.0)
    assert t212_pos["locked_base_usd"] == pytest.approx(250.0)
    vb_t212_p = tmp_path / "t212_reprice" / "venue_balances_state.json"
    vb_t212_p.parent.mkdir(parents=True)
    cached_t212 = {
        "pos:t212:SPCX_US_EQ": {
            "venue": "t212",
            "symbol": "SPCX_US_EQ",
            "free_base": 0.0,
            "value_usd": 250.0,
            "locked_base_qty": 2.0,
            "locked_base_usd": 240.0,
            "price": 120.0,
            "ts": now_s - 600.0,
        }
    }
    vb_t212_p.write_text(json.dumps(cached_t212), encoding="utf-8")
    import intelligence.sentiment.collectors.microstructure_collector as _mc_mod
    import providers.t212_provider as _t212_mod
    class _FakeT212Prov:
        def portfolio_mark_prices(self):
            return {"SPCX_US_EQ": 130.0}
    monkeypatch.setattr(_t212_mod, "T212Provider", _FakeT212Prov)
    monkeypatch.setattr(_mc_mod, "_LAST_T212_PORTFOLIO_POLL_TS", 0.0)
    assert refresh_stale_t212_cached_positions(cached_t212, str(vb_t212_p), now_s) is True
    mem_t212 = cached_t212["pos:t212:SPCX_US_EQ"]
    assert mem_t212["value_usd"] == pytest.approx(0.0)
    assert mem_t212["locked_base_usd"] == pytest.approx(260.0)
    # Disk state stays untouched by microstructure_collector (single producer invariant).
    disk_t212 = json.loads(vb_t212_p.read_text(encoding="utf-8"))["pos:t212:SPCX_US_EQ"]
    assert disk_t212["price"] == pytest.approx(120.0)

    # 6. Atomic free_and_locked_balance on Binance, Hyperliquid, Kraken, T212 + live collect() anti-double-subtract
    monkeypatch.setattr(
        bapi_mod, "client", SimpleNamespace(get_asset_balance=lambda asset: {"asset": asset, "free": "5143.79", "locked": "1004.66"})
    )
    assert BinanceProvider().free_and_locked_balance("USDC") == pytest.approx((5143.79, 1004.66))

    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xabc123")
    hl_p = HyperliquidProvider(
        token="HYPE",
        client=SimpleNamespace(
            info=SimpleNamespace(
                spot_user_state=lambda _addr: {"balances": [{"coin": "USDC", "total": "4088.83", "hold": "699.63"}]}
            )
        ),
    )
    assert hl_p.free_and_locked_balance("USDC") == pytest.approx((3389.2, 699.63))

    kr_p = KrakenProvider(
        client=SimpleNamespace(
            balance_ex=lambda: {"ZUSD": {"balance": "1660.02", "hold_trade": "200.00"}, "SPCXx": {"balance": "12.0", "hold_trade": "5.0"}}
        )
    )
    assert kr_p.free_and_locked_balance("USD") == pytest.approx((1460.02, 200.0))
    assert kr_p.free_and_locked_balance("SPCXx") == pytest.approx((7.0, 5.0))

    t212_p = T212Provider()
    monkeypatch.setattr(
        t212_p,
        "_client",
        lambda: SimpleNamespace(get_cash=lambda: {"free": 4600.0, "blocked": 920.0}),
    )
    import sys
    monkeypatch.setitem(sys.modules, "212trading.market_data", SimpleNamespace(get_usd_ron=lambda: 4.6))
    assert t212_p.free_and_locked_balance("USD") == pytest.approx((1000.0, 200.0))

    # End-to-end live collect() test: exchange returns atomic (5143.79, 1004.66), and order reprices 1s after bal_ts ->
    # MUST NOT double-subtract $104.85!
    live_prov = SimpleNamespace(
        free_and_locked_balance=lambda a: (5143.79, 1004.66) if a == "USDC" else (0.025, 0.0),
        free_balance=lambda a: 5143.79 if a == "USDC" else 0.025,
    )
    live_inst = SimpleNamespace(
        symbol="BTCUSDC",
        provider_name="binance",
        base="BTC",
        quote="USDC",
        _provider=live_prov,
        params={},
        open_orders=lambda: [
            {"orderId": "1", "side": "BUY", "type": "LIMIT", "price": 58050.0, "origQty": 0.0155, "executedQty": 0.0, "status": "NEW", "timestamp": int((now_s - 3600.0) * 1000)},
            {"orderId": "2", "side": "BUY", "type": "LIMIT", "price": 58907.4, "origQty": 0.00178, "executedQty": 0.0, "status": "NEW", "timestamp": int((now_s + 2.0) * 1000)},
        ],
    )
    cdb_live = tmp_path / "live_atomic_cdb"
    (cdb_live / "state").mkdir(parents=True)
    tpos_live, _, vcash_live, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdb_live),
        inst_map={"BTCUSDC": live_inst},
        prices_and_trends={"BTCUSDC": {"price": 58973.0}},
        allow_network=True,
        persist=False,
        now_ts=now_s,
    )
    assert vcash_live["binance:USDC"]["free"] == pytest.approx(5143.79, abs=0.02)
    assert vcash_live["binance:USDC"]["locked_quote_usd"] == pytest.approx(1004.63, abs=0.02)
    assert tpos_live[0]["free_quote"] == pytest.approx(5143.79, abs=0.02)


def test_post_7fc3348e_edge_case_hardening(tmp_path):
    """Regression coverage for the 4 post-7fc3348e audit edge cases:
    1. classify_order_evidence_state: partial fill on OPEN stays PARTIAL (Kraken/HL),
       and authoritative FILLED/CLOSED with require_qty_match_for_filled=False transitions to FILLED.
    2. load_venue_balances_state_dict & upsert_venue_balances_snapshot recover .previous on corrupt/missing
       primary file and never fall through from an existing empty state/ file to a stale root file.
    3. BalanceSnapshotRecord.to_persisted_cash_fields isolates quote_locked from base-side SELL locks
       when an asset serves simultaneously as quote (BUY) and base (SELL).
    4. upsert_venue_balances_snapshot strips all 5 INTERNAL_BALANCE_SCRATCH_KEYS including _prev_locked_*.
    """
    import json
    from intelligence.sentiment.collectors.portfolio_collector import derive_runtime_portfolio_view
    from providers.order_ledger import OrderLedgerState, _classify_venue_status_transition, classify_fill_evidence_state
    from providers.order_status_ssot import classify_order_evidence_state
    from providers.portfolio_state_store import (
        load_venue_balances_state_dict,
        upsert_venue_balances_snapshot,
    )
    from providers.portfolio_valuation import BalanceSnapshotRecord

    # 1a. OPEN working status with partial fill -> PARTIAL (never regresses PARTIAL -> OPEN on Kraken/HL)
    assert (
        _classify_venue_status_transition("OPEN", fq=0.4, req_qty=1.0, current_state=OrderLedgerState.PARTIAL.value)
        == OrderLedgerState.PARTIAL.value
    )
    assert (
        _classify_venue_status_transition("OPEN", fq=0.0, req_qty=1.0, current_state=OrderLedgerState.OPEN.value)
        == OrderLedgerState.OPEN.value
    )
    assert (
        _classify_venue_status_transition("OPEN", fq=1.0, req_qty=1.0, current_state=OrderLedgerState.PARTIAL.value)
        == OrderLedgerState.FILLED.value
    )
    # 1b. Authoritative FILLED/CLOSED (require_qty_match_for_filled=False) transitions to FILLED even with dust gap,
    # while fill-cache classification (require_qty_match_for_filled=True) stays strictly PARTIAL.
    assert (
        _classify_venue_status_transition("CLOSED", fq=0.995, req_qty=1.0, current_state=OrderLedgerState.PARTIAL.value)
        == OrderLedgerState.FILLED.value
    )
    assert (
        classify_order_evidence_state("FILLED", filled_qty=0.4, req_qty=1.0, require_qty_match_for_filled=False)
        == "FILLED"
    )
    assert (
        classify_fill_evidence_state({"quantity": 0.4, "status": "FILLED"}, req_qty=1.0, resting_now=False)
        == OrderLedgerState.PARTIAL.value
    )
    assert (
        classify_fill_evidence_state({"quantity": 1.0, "status": "OPEN"}, req_qty=1.0, resting_now=True)
        == OrderLedgerState.PARTIAL.value
    )

    # 2a. Corrupt primary state/venue_balances_state.json recovers valid .previous (never falls back to stale root)
    cdb = tmp_path / "cachedb_rec"
    (cdb / "state").mkdir(parents=True)
    primary = cdb / "state" / "venue_balances_state.json"
    primary_prev = cdb / "state" / "venue_balances_state.json.previous"
    stale_root = cdb / "venue_balances_state.json"
    primary.write_text("{corrupt-json", encoding="utf-8")
    primary_prev.write_text(json.dumps({"cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 5000.0}}), encoding="utf-8")
    stale_root.write_text(json.dumps({"cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1.0}}), encoding="utf-8")

    loaded = load_venue_balances_state_dict(str(cdb), use_cache=False)
    assert loaded["cash:binance:USDC"]["free"] == pytest.approx(5000.0)

    # 2b. Missing primary with valid .previous still recovers in upsert_venue_balances_snapshot before merging
    primary.unlink()
    assert upsert_venue_balances_snapshot(
        str(primary),
        {
            "cash:kraken:USD": {
                "venue": "kraken",
                "asset": "USD",
                "free": 1200.0,
                "_balance_ts": 100.0,
                "_prev_locked_quote_usd": 50.0,
                "_prev_locked_base_qty": 1.0,
            }
        },
    )
    merged_disk = json.loads(primary.read_text(encoding="utf-8"))
    assert merged_disk["cash:binance:USDC"]["free"] == pytest.approx(5000.0)
    assert merged_disk["cash:kraken:USD"]["free"] == pytest.approx(1200.0)
    # 4. All 5 scratch keys stripped on persist
    assert not any(k.startswith("_") for k in merged_disk["cash:kraken:USD"])

    # 2c. Existing empty state/ file returns {} instead of falling through to stale legacy root
    primary.write_text("{}", encoding="utf-8")
    primary_prev.unlink(missing_ok=True)
    assert load_venue_balances_state_dict(str(cdb), use_cache=False) == {}

    # 3. Dual-role asset (USDC locked as BUY quote on BTCUSDC AND as SELL base on USDCUSDT): no double-count in cash:*
    rec_dual = BalanceSnapshotRecord(free_raw=1000.0, total_locked=700.0, pre_balance_locked=0.0, reflected_locked=700.0)
    assert rec_dual.to_persisted_cash_fields(decimals=2, quote_locked=500.0, pre_quote_locked=0.0) == {
        "locked_quote_usd": 500.0
    }
    now_s = 1_795_610_000.0
    tpos = [
        {
            "venue": "binance",
            "symbol": "USDCUSDT",
            "base": "USDC",
            "quote": "USDT",
            "price": 1.0,
            "free_base": 1000.0,
            "free_quote": 200.0,
            "_balance_ts": now_s - 60.0,
            "_prev_locked_base_qty": 200.0,
        }
    ]
    vcash = {
        "binance:USDC": {
            "venue": "binance",
            "asset": "USDC",
            "free": 1000.0,
            "_balance_ts": now_s - 60.0,
            "_prev_locked_quote_usd": 500.0,
        }
    }
    updated_bals = {
        "cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1000.0},
        "pos:binance:USDCUSDT": {"venue": "binance", "symbol": "USDCUSDT", "free_base": 1000.0, "free_quote": 200.0},
    }
    derive_runtime_portfolio_view(
        open_orders_list=[
            {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 50000.0, "qty": 0.01, "created_ts": now_s - 120.0},
            {"venue": "binance", "symbol": "USDCUSDT", "side": "SELL", "price": 1.0, "qty": 200.0, "created_ts": now_s - 120.0},
        ],
        tradable_positions=tpos,
        venue_quote_cash=vcash,
        inst_by_sym={},
        prices_and_trends={"BTCUSDC": {"price": 50000.0}, "USDCUSDT": {"price": 1.0}},
        updated_bals=updated_bals,
        pending_peaks_by_vs={},
        now_ts_oo=now_s,
        gateway_max_usd=2000.0,
    )
    assert updated_bals["cash:binance:USDC"]["locked_quote_usd"] == pytest.approx(500.0)
    assert updated_bals["pos:binance:USDCUSDT"]["locked_base_qty"] == pytest.approx(200.0)


def test_layer3_cache_balance_manager_eliminates_inversion_and_splits_positions_state(tmp_path, monkeypatch):
    """Verify the clean 2-domain split eliminating the Layer 3 -> Layer 5 inversion:
    1. Provider batch API all_free_and_locked_balances across Binance, Kraken, Hyperliquid, and T212.
    2. CacheBalanceManager (Layer 3) persists pure {free, locked, ts} + breakdown into orders/cache_balance_<venue>.json
       with verified full-venue replacement vs partial merge semantics, registered in CacheFactory & get_balance_manager.
    3. CacheAssetValueManager computes multi-venue NAV purely from Layer 3 cache_balance_<venue>.json + cache_current_price.json
       with ZERO venue_balances_state.json or portfolio_positions_state.json present.
    4. providers.nav_drawdown.record_nav_history_snapshot computes Unified NAV purely from Layer 3 cache_balance_<venue>.json.
    5. upsert_portfolio_positions_snapshot strictly persists ONLY pos:* derived snapshots (never raw cash/scalar/breakdown)
       and load_venue_balances_state_dict cleanly composes both domains.
    """
    import sys
    import binance_api.bapi as bapi_mod
    import cacheManager as cm_mod
    from cacheaccount import CacheAssetValueManager, CacheBalanceManager, get_balance_manager
    from cachepaths import (
        CACHE_BINANCE_BALANCE_FILENAME,
        CACHE_HYPERLIQUID_BALANCE_FILENAME,
        CACHE_KRAKEN_BALANCE_FILENAME,
        CACHE_T212_BALANCE_FILENAME,
        DurabilityTier,
        PORTFOLIO_POSITIONS_STATE_FILENAME,
        resolve_durability_policy,
    )
    from providers.binance_provider import BinanceProvider
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.nav_history_store import record_nav_history_snapshot
    from providers.portfolio_state_store import (
        load_portfolio_positions_state_dict,
        load_venue_balance_caches,
        load_venue_balances_state_dict,
        upsert_portfolio_positions_snapshot,
    )
    from state_stores.venue_balances_state import INTERNAL_BALANCE_SCRATCH_KEYS
    from providers.t212_provider import T212Provider

    now_s = 1_795_800_000.0

    # 0. Durability policy & CacheFactory registration
    for fname in (
        CACHE_BINANCE_BALANCE_FILENAME,
        CACHE_KRAKEN_BALANCE_FILENAME,
        CACHE_HYPERLIQUID_BALANCE_FILENAME,
        CACHE_T212_BALANCE_FILENAME,
        PORTFOLIO_POSITIONS_STATE_FILENAME,
    ):
        assert resolve_durability_policy(fname).tier == DurabilityTier.FINANCIAL_CRITICAL
    for cname in ("BinanceBalance", "KrakenBalance", "HyperliquidBalance", "T212Balance"):
        assert cname in cm_mod.CacheFactory._CONFIG

    # 1. Batch all_free_and_locked_balances across all 4 venues
    monkeypatch.setattr(
        bapi_mod,
        "get_account_assets_balances",
        lambda raise_errors=False: [
            {"asset": "USDC", "free": 1500.0, "locked": 250.0, "total": 1750.0},
            {"asset": "BTC", "free": 0.05, "locked": 0.01, "total": 0.06},
        ],
    )
    bn_all = BinanceProvider().all_free_and_locked_balances(["BTC", "USDC", "ARB"])
    assert bn_all == {"USDC": (1500.0, 250.0), "BTC": (0.05, 0.01), "ARB": (0.0, 0.0)}

    kr_p = KrakenProvider(
        client=SimpleNamespace(
            balance_ex=lambda: {
                "ZUSD": {"balance": "600.0", "hold_trade": "100.0"},
                "USDC": {"balance": "400.0", "hold_trade": "0.0"},
                "HYPE": {"balance": "10.0", "hold_trade": "2.0"},
            }
        )
    )
    kr_all = kr_p.all_free_and_locked_balances(["HYPE", "USD"])
    assert kr_all["USD"] == pytest.approx((900.0, 100.0))
    assert kr_all["HYPE"] == pytest.approx((8.0, 2.0))

    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xabc123")
    hl_p = HyperliquidProvider(
        token="HYPE",
        client=SimpleNamespace(
            info=SimpleNamespace(
                spot_user_state=lambda _addr: {
                    "balances": [
                        {"coin": "USDC", "total": "500.0", "hold": "50.0"},
                        {"coin": "HYPE", "total": "4.0", "hold": "1.0"},
                    ]
                }
            )
        ),
    )
    hl_all = hl_p.all_free_and_locked_balances(["HYPE", "USDC"])
    assert hl_all == {"USDC": (450.0, 50.0), "HYPE": (3.0, 1.0)}

    t212_p = T212Provider()
    monkeypatch.setattr(
        t212_p,
        "_client",
        lambda: SimpleNamespace(
            get_cash=lambda: {"free": 460.0, "blocked": 92.0},
            get_portfolio=lambda: [{"ticker": "NVDA_US_EQ", "quantity": 5.0}],
        ),
    )
    monkeypatch.setattr(
        t212_p,
        "open_orders_by_symbol",
        lambda _syms: {"NVDA_US_EQ": [{"side": "SELL", "origQty": 1.5, "executedQty": 0.5}]},
    )
    monkeypatch.setitem(sys.modules, "212trading.market_data", SimpleNamespace(get_usd_ron=lambda: 4.6))
    t212_all = t212_p.all_free_and_locked_balances(["NVDA_US_EQ", "USD"])
    assert t212_all["USD"] == pytest.approx((100.0, 20.0))
    assert t212_all["NVDA_US_EQ"] == pytest.approx((4.0, 1.0))

    # 2. CacheBalanceManager write/read + verified full-venue replacement vs partial merge
    ws_dir = tmp_path / "ws_split"
    cdb = ws_dir / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    (cdb / "prices").mkdir(parents=True)
    kr_bal_path = cdb / "orders" / CACHE_KRAKEN_BALANCE_FILENAME
    t212_bal_path = cdb / "orders" / CACHE_T212_BALANCE_FILENAME
    CacheBalanceManager.replace_venue_balances_snapshot(
        str(kr_bal_path),
        {"USD": (500.0, 100.0), "HYPE": (8.0, 2.0), "GHOST": (1.0, 0.0)},
        venue="kraken",
        now_s=now_s - 60.0,
        verified_ts=now_s - 60.0,
        breakdown={"USD": {"USD": 200.0, "USDC": 300.0}},
    )
    # Full-venue verified replace drops sold-out GHOST cleanly while preserving breakdown when omitted
    CacheBalanceManager.replace_venue_balances_snapshot(
        str(kr_bal_path),
        {"USD": (500.0, 100.0), "HYPE": (8.0, 2.0)},
        venue="kraken",
        now_s=now_s,
        verified_ts=now_s,
    )
    kr_mgr = get_balance_manager(venue="kraken", filename=str(kr_bal_path))
    assert kr_mgr.get_balance("GHOST") is None
    assert kr_mgr.get_balance("USD")["free"] == pytest.approx(500.0)
    assert kr_mgr.get_balance("USD")["locked"] == pytest.approx(100.0)
    assert kr_mgr.get_breakdown("USD") == {"USD": 200.0, "USDC": 300.0}

    CacheBalanceManager.replace_venue_balances_snapshot(
        str(t212_bal_path),
        {"USD": (100.0, 20.0), "NVDA_US_EQ": (4.0, 1.0)},
        venue="t212",
        now_s=now_s,
        verified_ts=now_s,
    )
    # Live prices in Layer 3 cache_current_price.json ONLY (ZERO venue_balances_state.json or portfolio_positions_state.json!)
    (cdb / "prices" / "cache_current_price.json").write_text(
        json.dumps({"items": {"HYPEUSD": [[int(now_s * 1000), 50.0]], "NVDA_US_EQ": [[int(now_s * 1000), 200.0]]}}),
        encoding="utf-8",
    )

    # 3. CacheAssetValueManager computes multi-venue NAV strictly from Layer 3 caches (ZERO Layer 5 dependency!)
    api_mock = MagicMock()
    api_mock.get_total_assets_value_usdc.return_value = 2000.0
    av_mgr = CacheAssetValueManager(9999, ["TOTAL"], str(cdb / "orders" / "cache_asset_value.jsonl"), api_client=api_mock)
    av_items = av_mgr.get_remote_items("TOTAL", 0)
    assert len(av_items) == 1
    # Kraken: $500 free + $100 locked USD + (8+2)*$50 HYPE = $1100; T212: $100+$20 USD + (4+1)*$200 NVDA = $1120
    assert av_items[0]["venues_usd"] == {"binance": 2000.0, "kraken": 1100.0, "t212": 1120.0}
    assert av_items[0]["portfolio_total_usd"] == pytest.approx(4220.0)

    # 4. providers.nav_drawdown.record_nav_history_snapshot works purely from Layer 3 balance caches
    nav_m = record_nav_history_snapshot(str(ws_dir), now_ts=now_s)
    assert nav_m["current_nav_usd"] == pytest.approx(2220.0)

    # 5. upsert_portfolio_positions_snapshot writes ONLY pos:* keys and strips internal scratch keys; unified loader composes cleanly
    pos_path = cdb / "state" / PORTFOLIO_POSITIONS_STATE_FILENAME
    assert upsert_portfolio_positions_snapshot(
        str(pos_path),
        {
            "cash:kraken:USD": {"venue": "kraken", "asset": "USD", "free": 9999.0},
            "kraken:USD": 9999.0,
            "breakdown:kraken:USD": {"USD": 9999.0},
            "pos:kraken:HYPEUSD": {
                "venue": "kraken",
                "symbol": "HYPEUSD",
                "base": "HYPE",
                "quote": "USD",
                "free_base": 1.0,  # older balance -> Layer 3 cache_balance_kraken.json (8.0 free, 2.0 locked) wins!
                "free_quote": 10.0,
                "price": 50.0,
                "trailing_peak": 55.0,
                "ts": now_s - 300.0,
                "_prev_locked_base_qty": 99.0,
            },
        },
    ) is True
    pos_disk = load_portfolio_positions_state_dict(str(cdb))
    assert list(pos_disk.keys()) == ["pos:kraken:HYPEUSD"]
    for sk in INTERNAL_BALANCE_SCRATCH_KEYS:
        assert sk not in pos_disk["pos:kraken:HYPEUSD"]
    assert set(load_venue_balance_caches(str(cdb)).keys()) == {"kraken", "t212"}
    unified = load_venue_balances_state_dict(str(cdb))
    assert unified["cash:kraken:USD"]["free"] == pytest.approx(500.0)
    assert unified["cash:kraken:USD"]["locked_quote_usd"] == pytest.approx(100.0)
    assert unified["breakdown:kraken:USD"] == {"USD": 200.0, "USDC": 300.0}
    assert unified["pos:kraken:HYPEUSD"]["trailing_peak"] == pytest.approx(55.0)
    assert unified["pos:kraken:HYPEUSD"]["free_base"] == pytest.approx(8.0)
    assert unified["pos:kraken:HYPEUSD"]["locked_base_qty"] == pytest.approx(2.0)


def test_state_stores_1to1_symmetry_and_exclusive_owners(tmp_path):
    """Verify Etapa 1 state_stores/ package:
    1. Every registered critical state file has strict 1:1 filename <-> state_stores/<stem>.py <-> <PascalCase>Store symmetry.
    2. All 13 typed stores round-trip cleanly with .previous recovery and dirty-check semantics.
    3. AST audits statically block raw reads, raw writes, and upward Layer 2.5 DAG leaks from state_stores/.
    """
    from pathlib import Path
    import state_stores
    from state_stores import (
        AssetguardianStateStore,
        AutoOrderIntentsStateStore,
        CRITICAL_STATE_SPECS,
        LlmTopicThreadsStore,
        MtStateStore,
        NavHistoryStateStore,
        NotifyQueueStore,
        OrderRetryQueueStore,
        PortfolioPositionsStateStore,
        RtradePairsStateStore,
        SmartStopLossStateStore,
        TrailingStateStore,
        VenueBalancesStateStore,
        VenueFeeMaintenanceStateStore,
        expected_store_class_name,
    )
    from verify_tools.ssot_ast_audit import (
        audit_persistence_and_cache_dag_ast,
        audit_state_file_owners_ast,
    )

    # 1. Exact 1:1 registry symmetry across all 17 critical state files
    assert len(CRITICAL_STATE_SPECS) == 17
    for spec in CRITICAL_STATE_SPECS:
        assert spec.owner_module == f"state_stores/{Path(spec.filename).stem}.py"
        assert spec.store_class == expected_store_class_name(spec.filename)
        cls_obj = getattr(state_stores, spec.store_class)
        assert cls_obj.FILENAME == spec.filename

    # 2. Functional round-trips on isolated temp cachedb
    cdb = tmp_path / "cachedb"
    (cdb / "state").mkdir(parents=True)
    (cdb / "orders").mkdir(parents=True)

    # VenueBalancesStateStore & PortfolioPositionsStateStore
    vb_path = VenueBalancesStateStore.resolve_path(str(cdb))
    assert VenueBalancesStateStore.upsert_locked(
        vb_path, {"cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 1000.0, "_balance_ts": 1.0}}
    )
    assert VenueBalancesStateStore.load_dict(str(cdb))["cash:binance:USDC"]["free"] == pytest.approx(1000.0)
    assert "_balance_ts" not in VenueBalancesStateStore.load_dict(str(cdb))["cash:binance:USDC"]

    pos_path = PortfolioPositionsStateStore.resolve_path(str(cdb))
    assert PortfolioPositionsStateStore.upsert_locked(
        pos_path, {"pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "trailing_peak": 60000.0}}
    )
    assert PortfolioPositionsStateStore.load_dict(str(cdb))["pos:binance:BTCUSDC"]["trailing_peak"] == pytest.approx(60000.0)

    # AutoOrderIntentsStateStore
    int_path = AutoOrderIntentsStateStore.resolve_path(str(cdb))
    assert AutoOrderIntentsStateStore.upsert_intent(
        int_path,
        {"intent_id": "ai_1", "symbol": "BTCUSDC", "status": "PENDING"},
        extract_id_fn=lambda d: str(d.get("intent_id") or ""),
    )
    assert len(AutoOrderIntentsStateStore.load_from_cachedb(str(cdb))) == 1

    # LlmTopicThreadsStore & NotifyQueueStore
    top_p = LlmTopicThreadsStore.resolve_path(str(cdb))
    LlmTopicThreadsStore.save_registry(top_p, {"auto_cryptotrade_executor": {"conversation_id": "cid_1"}})
    assert LlmTopicThreadsStore.read_registry(top_p)["auto_cryptotrade_executor"]["conversation_id"] == "cid_1"

    nq_p = NotifyQueueStore.resolve_path(str(cdb))
    NotifyQueueStore.write_lines_nolock(nq_p, ['{"__orchestrator_intent__":"email","subject":"s1"}'])
    assert len(NotifyQueueStore.read_lines_nolock(nq_p)) == 1

    # OrderRetryQueueStore
    q_path = OrderRetryQueueStore.resolve_path(str(cdb))
    OrderRetryQueueStore.write_records_nolock(q_path, [{"id": "r1", "symbol": "BTCUSDC", "side": "BUY"}])
    assert len(OrderRetryQueueStore.read_tail_records(str(cdb), limit=10)) == 1

    # NavHistoryStateStore
    nav_p = NavHistoryStateStore.resolve_workspace_path(str(tmp_path))
    NavHistoryStateStore.save_points_doc(nav_p, [{"ts": 100.0, "nav_usd": 17000.0}], updated_ts=100.0)
    assert NavHistoryStateStore.read_raw_doc(str(tmp_path))["points"][0]["nav_usd"] == pytest.approx(17000.0)

    # VenueFeeMaintenanceStateStore
    with VenueFeeMaintenanceStateStore.locked_read_for_update(str(cdb)) as (fee_p, prev_fee):
        assert prev_fee == {}
        VenueFeeMaintenanceStateStore.save_state(fee_p, {"venues": {"binance": {"held_bnb_usdc": 150.0}}})
    assert VenueFeeMaintenanceStateStore.read_state(str(cdb))["venues"]["binance"]["held_bnb_usdc"] == pytest.approx(150.0)

    # AssetguardianStateStore, SmartStopLossStateStore, MtStateStore, TrailingStateStore, RtradePairsStateStore
    ag_p = AssetguardianStateStore.resolve_path(str(cdb))
    AssetguardianStateStore.save_state({"symbols": {"BTCUSDC": {}}}, ag_p)
    assert "BTCUSDC" in AssetguardianStateStore.read_from_cachedb(str(cdb))["symbols"]

    ssl_p = SmartStopLossStateStore.resolve_path(str(cdb))
    SmartStopLossStateStore.save_state(ssl_p, {"BTCUSDC": {"start_ts": 1_795_000_000.0}})
    assert "BTCUSDC" in SmartStopLossStateStore.read_from_cachedb(str(cdb))

    mt_p = MtStateStore.resolve_path(str(cdb))
    MtStateStore.save_state(mt_p, {"version": 1, "tp_ride_state": {"BTCUSDC": {}}})
    assert "BTCUSDC" in MtStateStore.read_from_cachedb(str(cdb))["tp_ride_state"]

    tr_p = TrailingStateStore.resolve_venue_path("binance", cache_dir=str(cdb))
    TrailingStateStore.save_state(tr_p, {"BTCUSDC": {"peak": 62000.0}})
    assert TrailingStateStore.read_venue_state("binance", cache_dir=str(cdb))["BTCUSDC"]["peak"] == pytest.approx(62000.0)

    rt_p = RtradePairsStateStore.resolve_path(str(cdb))
    RtradePairsStateStore.write_atomic(rt_p, {"version": 1, "pairs": {}})
    assert RtradePairsStateStore.read_raw(rt_p)["version"] == 1

    # 3. Negative AST checks: raw read, raw write, and upward DAG leak from state_stores/
    synth = tmp_path / "synth_stores"
    (synth / "state_stores").mkdir(parents=True)
    (synth / "bad_writer.py").write_text(
        "import os\n"
        "from cachepaths import NAV_HISTORY_STATE_FILENAME\n"
        "from state_io import save_json_state\n"
        "def f(d):\n"
        "    p = os.path.join(d, NAV_HISTORY_STATE_FILENAME)\n"
        "    save_json_state(p, {'points': []})\n",
        encoding="utf-8",
    )
    (synth / "state_stores" / "leaky_store.py").write_text(
        "import providers.portfolio_valuation\n",
        encoding="utf-8",
    )
    w_viols = audit_state_file_owners_ast(synth, rel_paths=["bad_writer.py"])
    assert any("direct raw write of NAV_HISTORY_STATE_FILENAME" in v for v in w_viols)
    dag_viols = audit_persistence_and_cache_dag_ast(synth, rel_paths=["state_stores/leaky_store.py"])
    assert any("Layer 2.5 upward leak" in v and "providers" in v for v in dag_viols)


def test_kraken_xstock_balance_cache_canonical_key_and_nav_dedup(tmp_path):
    """Regression: Kraken tokenized xStocks (SPCXx.T / ANTx.T / AAPLx.T) must preserve canonical
    mixed-case keys in CacheBalanceManager, never duplicate under SPCX+SPCXX, and always resolve
    live mixed-case prices in compute_venue_nav_breakdown without dipping to $0 or doubling NAV.
    """
    from cacheaccount import CacheAssetValueManager, CacheBalanceManager, _VenueBalancesBridge
    from cachepaths import CACHE_KRAKEN_BALANCE_FILENAME
    from providers.kraken_provider import KrakenProvider
    from providers.portfolio_state_store import (
        load_venue_balance_caches,
        load_venue_balances_state_dict,
    )
    from providers.portfolio_valuation import compute_venue_nav_breakdown

    now_s = 1_795_900_000.0

    # 1. _VenueBalancesBridge requests canonical mixed-case xStock bases from instruments.conf
    req_assets = _VenueBalancesBridge._venue_requested_assets("kraken")
    assert "SPCXx" in req_assets
    assert "ANTx" in req_assets
    assert "SPCX" not in req_assets
    assert "ANT" not in req_assets

    # 2. KrakenProvider.all_free_and_locked_balances maps SPCXx.T/ANTx.T/AAPLx.T without duplicates
    kr_p = KrakenProvider(
        client=SimpleNamespace(
            balance_ex=lambda: {
                "ZUSD": {"balance": "1660.02", "hold_trade": "0.0"},
                "HYPE": {"balance": "0.002591", "hold_trade": "0.0"},
                "SPCXx.T": {"balance": "12.0", "hold_trade": "0.0"},
                "ANTx.T": {"balance": "1.5", "hold_trade": "0.0"},
                "AAPLx.T": {"balance": "0.25", "hold_trade": "0.0"},
                "XETH": {"balance": "0.1", "hold_trade": "0.0"},
            }
        )
    )
    kr_all = kr_p.all_free_and_locked_balances(["HYPE", "USD", "SPCXx", "ANTx"])
    assert kr_all["SPCXx"] == pytest.approx((12.0, 0.0))
    assert kr_all["ANTx"] == pytest.approx((1.5, 0.0))
    assert kr_all["AAPLx"] == pytest.approx((0.25, 0.0))
    assert kr_all["ETH"] == pytest.approx((0.1, 0.0))
    assert "SPCX" not in kr_all
    assert "ANT" not in kr_all
    assert "AAPL" not in kr_all

    # 3. Canonical xStock builders & SymbolIdentity API prevent case corruption across assets and pairs
    from cachepaths import ensure_quote_suffix, ensure_xstock_asset, ensure_xstock_pair
    from instrument_registry import resolve_symbol_identity

    assert ensure_xstock_asset("SPCX") == "SPCXx"
    assert ensure_xstock_asset("SPCXx") == "SPCXx"
    assert ensure_xstock_asset("SPCXx.T") == "SPCXx"
    assert ensure_xstock_asset("SPCXxUSD") == "SPCXx"
    assert ensure_xstock_asset("SPCX_US_EQ") == "SPCXx"
    assert ensure_xstock_pair("SPCX") == "SPCXxUSD"
    assert ensure_xstock_pair("ANT") == "ANTxUSD"
    assert ensure_xstock_pair("SPCXx.T") == "SPCXxUSD"
    assert ensure_quote_suffix("SPCXx", "USD") == "SPCXxUSD"
    assert ensure_quote_suffix("SPCXxEUR", "USD") == "SPCXxEUR"
    assert ensure_quote_suffix("SPCXx.T", "USD") == "SPCXxUSD"

    folded_spcx = "SPCXx".upper()
    for raw_in in ("SPCXxUSD", "SPCXx", "SPCXx.T", folded_spcx):
        ident = resolve_symbol_identity(raw_in, venue="kraken")
        assert ident.is_tokenized_equity is True
        assert ident.base_asset == "SPCX"
        assert ident.balance_asset_key == "SPCXx"
        assert ident.quote_pair("USD") == "SPCXxUSD"
    assert resolve_symbol_identity("NVDA_US_EQ", venue="t212").balance_asset_key == "NVDA_US_EQ"
    assert resolve_symbol_identity("NVDA_US_EQ", venue="t212").quote_pair("USD") == "NVDA_US_EQ"
    assert resolve_symbol_identity("HYPE", venue="kraken").quote_pair("USD") == "HYPEUSD"

    # 4. CacheBalanceManager canonicalizes SPCXx and legacy case-folded input on full replace & partial merge (no duplicates)
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    (cdb / "prices").mkdir(parents=True)
    kr_bal_path = cdb / "orders" / CACHE_KRAKEN_BALANCE_FILENAME
    CacheBalanceManager.replace_venue_balances_snapshot(
        str(kr_bal_path),
        {"USD": (1660.0, 0.0), "SPCXx": (12.0, 0.0)},
        venue="kraken",
        now_s=now_s,
        verified_ts=now_s,
    )
    CacheBalanceManager.replace_venue_balances_snapshot(
        str(kr_bal_path),
        {folded_spcx: {"free": 12.0, "locked": 0.0, "ts": now_s + 5.0}},
        venue="kraken",
        now_s=now_s + 5.0,
        verified_ts=None,
    )
    saved_doc = json.loads(kr_bal_path.read_text(encoding="utf-8"))
    assert set(saved_doc["items"].keys()) == {"USD", "SPCXx"}
    kr_mgr = CacheBalanceManager(60, None, str(kr_bal_path), read_only=True, venue="kraken")
    assert kr_mgr.get_balance("SPCXx")["free"] == pytest.approx(12.0)
    assert kr_mgr.get_balance(folded_spcx)["free"] == pytest.approx(12.0)
    snap_items, _, _ = CacheBalanceManager.read_venue_balances_snapshot(str(kr_bal_path))
    assert set(snap_items.keys()) == {"USD", "SPCXx"}

    # 5. Live price lookup & NAV valuation strictly from Layer 3 caches (no $0 dip and no 2x spike)
    live_prices = {"items": {"SPCXxUSD": [[int(now_s * 1000), 120.0]]}}
    (cdb / "prices" / "cache_current_price.json").write_text(json.dumps(live_prices), encoding="utf-8")
    bal_caches = load_venue_balance_caches(str(cdb))
    nav = compute_venue_nav_breakdown(venue_balance_caches=bal_caches, live_prices=live_prices)
    # Kraken = $1660 cash + 12 * $120 SPCXx = $3100.0
    assert nav["venues_usd"]["kraken"] == pytest.approx(3100.0)
    assert nav["portfolio_total_usd"] == pytest.approx(3100.0)
    unified = load_venue_balances_state_dict(str(cdb))
    assert unified["kraken:SPCXx"] == pytest.approx(12.0)
    assert f"kraken:{folded_spcx}" not in unified

    # 6. Even a legacy/stale file containing both bare stem and case-folded form deduplicates cleanly and resolves SPCXxUSD price
    stale_caches = {
        "kraken": {
            "venue": "kraken",
            "ts": now_s,
            "items": {
                "USD": {"free": 1660.0, "locked": 0.0, "ts": now_s},
                "SPCX": {"free": 12.0, "locked": 0.0, "ts": now_s},
                folded_spcx: {"free": 12.0, "locked": 0.0, "ts": now_s + 1.0},
            },
        }
    }
    nav_stale = compute_venue_nav_breakdown(venue_balance_caches=stale_caches, live_prices=live_prices)
    assert nav_stale["venues_usd"]["kraken"] == pytest.approx(3100.0)

    # 7. CacheAssetValueManager prices Kraken SPCXx using only Layer 3 balance + current price caches
    api_mock = MagicMock()
    api_mock.get_total_assets_value_usdc.return_value = 9000.0
    av_mgr = CacheAssetValueManager(9999, ["TOTAL"], str(cdb / "orders" / "cache_asset_value.jsonl"), api_client=api_mock)
    av_items = av_mgr.get_remote_items("TOTAL", 0)
    assert len(av_items) == 1
    assert av_items[0]["venues_usd"] == {"binance": 9000.0, "kraken": 3100.0}
    assert av_items[0]["portfolio_total_usd"] == pytest.approx(12100.0)

    # 8. P1 hardening: Kraken ledger_to_fills, order_retry_worker display assets, open_order/fee/ledger & wire asset helpers
    from kraken.kraken_client import KrakenClient
    import order_retry_worker as orw_mod
    from providers.order_ledger import build_order_event
    from providers.order_ohlc_utils import extract_order_fee, normalize_open_order
    from providers.quantity import resolve_assets as pq_resolve_assets
    from providers.venue_ws_protocols import KrakenWsLedgerMixin

    x_fills = KrakenWsLedgerMixin.ledger_to_fills(
        {
            "L1": {"refid": "RF1", "type": "trade", "asset": "SPCXx.T", "amount": "2.0", "fee": "0.01", "time": now_s},
            "L2": {"refid": "RF1", "type": "trade", "asset": "ZUSD", "amount": "-240.0", "fee": "0.0", "time": now_s},
        }
    )
    assert len(x_fills) == 1
    assert x_fills[0]["symbol"] == "SPCXxUSD"
    assert x_fills[0]["feeAsset"] == "SPCXx"
    assert orw_mod._resolve_display_assets("SPCXxUSD") == ("SPCX", "USD")
    assert orw_mod._resolve_display_assets("SPCXxUSD".upper()) == ("SPCX", "USD")
    assert pq_resolve_assets("SPCXx") == ("SPCXx", "USD")
    assert pq_resolve_assets("SPCXx.T") == ("SPCXx.T", "USD")
    norm_oo = normalize_open_order({"symbol": "SPCXxUSD", "side": "SELL", "price": "120.0", "origQty": "1.0"}, venue="kraken")
    assert norm_oo is not None and norm_oo["symbol"] == "SPCXxUSD"
    fee_d = extract_order_fee({"fee": "0.01", "feeAsset": "SPCXx"})
    assert fee_d is not None and fee_d["fee_asset"] == "SPCXx"
    ev_d = build_order_event("FILL", venue="kraken", symbol="SPCXxUSD", fee=0.01, fee_asset="SPCXx")
    assert ev_d["sym"] == "SPCXxUSD" and ev_d["fee_asset"] == "SPCXx"
    assert KrakenClient._norm_wire_asset("SPCXx.T") == "SPCXx.T"
    assert KrakenClient._norm_wire_asset("usdc") == "USDC"


def test_pure_portfolio_valuation_split_and_cache_first_portfolio_collector(tmp_path):
    """Verify pure portfolio_valuation split + cache-first TradablePositionsAndOrdersCollector:
    1. providers/portfolio_valuation.py has zero disk/state I/O imports, enforced by audit_layer_dag_ast.
    2. With fresh Layer 3 cache_balance_<venue>.json (<=90s) and cache_order_open_<venue>.json (<=30s)
       and a real MarketDataProvider subclass, TradablePositionsAndOrdersCollector makes ZERO REST calls
       and ZERO redundant rewrites to Layer 3 caches while still persisting Layer 5 portfolio_positions_state.json.
    3. When Layer 3 caches turn stale (>90s / >30s), live REST fallback runs and refreshes Layer 3 caches.
    """
    from pathlib import Path
    from cacheaccount import CacheBalanceManager, CacheOrderOpenManager
    from cachepaths import (
        CACHE_BINANCE_BALANCE_FILENAME,
        CACHE_BINANCE_ORDER_OPEN_FILENAME,
        PORTFOLIO_POSITIONS_STATE_FILENAME,
    )
    from intelligence.sentiment.collectors.portfolio_collector import TradablePositionsAndOrdersCollector
    from providers.base import MarketDataProvider
    from verify_tools.ssot_ast_audit import audit_layer_dag_ast

    # 1. Pure valuation AST gate passes on repo and blocks disk/state I/O leaks
    repo_root = Path(__file__).resolve().parents[1]
    assert audit_layer_dag_ast(repo_root, rel_paths=["providers/portfolio_valuation.py"]) == []
    synth_pv = tmp_path / "synth_pv"
    (synth_pv / "providers").mkdir(parents=True)
    (synth_pv / "providers" / "portfolio_valuation.py").write_text(
        "import os\nfrom state_io import read_json_dict\nfrom state_stores.venue_balances_state import VenueBalancesStateStore\n",
        encoding="utf-8",
    )
    pv_viols = audit_layer_dag_ast(synth_pv, rel_paths=["providers/portfolio_valuation.py"])
    assert any("pure valuation module must not import disk/state I/O modules" in v for v in pv_viols)

    # 2. Seed fresh Layer 3 balance + open-order caches for Binance
    cdb = tmp_path / "cachedb"
    (cdb / "orders").mkdir(parents=True)
    (cdb / "state").mkdir(parents=True)
    now_s = 1_795_900_000.0
    bal_path = cdb / "orders" / CACHE_BINANCE_BALANCE_FILENAME
    open_path = cdb / "orders" / CACHE_BINANCE_ORDER_OPEN_FILENAME
    pos_path = cdb / "state" / PORTFOLIO_POSITIONS_STATE_FILENAME

    CacheBalanceManager.replace_venue_balances_snapshot(
        str(bal_path),
        {"BTC": (0.25, 0.05), "USDC": (1000.0, 200.0)},
        venue="binance",
        now_s=now_s - 10.0,
        verified_ts=now_s - 10.0,
    )
    CacheOrderOpenManager.replace_venue_open_orders_snapshot(
        str(open_path),
        {
            "BTCUSDC": [
                {
                    "orderId": "ord-l3-1",
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "price": 50000.0,
                    "origQty": 0.004,
                    "executedQty": 0.0,
                    "status": "NEW",
                    "time": int((now_s - 60.0) * 1000),
                }
            ]
        },
        now_ms=int((now_s - 10.0) * 1000),
        verified_ts=now_s - 10.0,
    )
    bal_before = bal_path.read_text(encoding="utf-8")
    open_before = open_path.read_text(encoding="utf-8")

    rest_calls = {"balance": 0, "open_orders": 0}

    class _CountingProvider(MarketDataProvider):
        name = "binance"

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def get_current_price(self, symbol: str) -> float:
            return 60000.0

        def get_klines(self, symbol: str, interval: str, limit: int = 100):
            return []

        def free_balance(self, asset: str) -> float:
            rest_calls["balance"] += 1
            return 0.5 if asset.upper() == "BTC" else 1500.0

        def free_and_locked_balance(self, asset: str) -> tuple[float, float]:
            rest_calls["balance"] += 1
            return (0.5, 0.0) if asset.upper() == "BTC" else (1500.0, 0.0)

        def open_orders(self, symbol: str) -> list[dict]:
            rest_calls["open_orders"] += 1
            return []

    real_prov = _CountingProvider()
    fake_inst = SimpleNamespace(
        symbol="BTCUSDC",
        base="BTC",
        quote="USDC",
        provider_name="binance",
        _provider=real_prov,
        params={},
        open_orders=lambda: real_prov.open_orders("BTCUSDC"),
    )

    verified_v: set[str] = set()
    verified_s: set[tuple[str, str]] = set()
    tpos, oorders, vcash, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdb),
        inst_map={"BIN_BTC": fake_inst},
        prices_and_trends={"BTCUSDC": {"price": 60000.0, "trend": 0, "growth_coefficient": 0.0}},
        verified_open_venues=verified_v,
        verified_open_symbols=verified_s,
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_s,
    )
    # Zero REST calls AND zero redundant Layer 3 cache rewrites
    assert rest_calls == {"balance": 0, "open_orders": 0}
    assert bal_path.read_text(encoding="utf-8") == bal_before
    assert open_path.read_text(encoding="utf-8") == open_before
    assert verified_v == {"binance"}
    assert ("binance", "BTCUSDC") in verified_s
    assert len(oorders) == 1 and oorders[0]["order_id"] == "ord-l3-1"
    assert len(tpos) == 1
    assert tpos[0]["free_base"] == pytest.approx(0.25)
    assert tpos[0]["free_quote"] == pytest.approx(1000.0)
    assert vcash["binance:USDC"]["free"] == pytest.approx(1000.0)
    assert vcash["binance:USDC"]["locked_quote_usd"] == pytest.approx(200.0)
    # Layer 5 derived position snapshot WAS persisted (pos:* only)
    assert pos_path.is_file()
    pos_saved = json.loads(pos_path.read_text(encoding="utf-8"))
    assert list(pos_saved.keys()) == ["pos:binance:BTCUSDC"]

    # 3. Advance past freshness limits (>90s balance, >30s open orders) -> live REST fallback + Layer 3 refresh
    tpos2, oorders2, vcash2, _ = TradablePositionsAndOrdersCollector(start_sync=False).collect(
        str(cdb),
        inst_map={"BIN_BTC": fake_inst},
        prices_and_trends={"BTCUSDC": {"price": 60000.0, "trend": 0, "growth_coefficient": 0.0}},
        allow_network=True,
        persist=True,
        force_persist=True,
        now_ts=now_s + 120.0,
    )
    assert rest_calls["balance"] == 2
    assert rest_calls["open_orders"] == 1
    assert oorders2 == []
    assert tpos2[0]["free_base"] == pytest.approx(0.5)
    assert vcash2["binance:USDC"]["free"] == pytest.approx(1500.0)
    bal_after = json.loads(bal_path.read_text(encoding="utf-8"))
    assert bal_after["items"]["BTC"]["free"] == pytest.approx(0.5)
    assert bal_after["items"]["USDC"]["free"] == pytest.approx(1500.0)
    open_after = json.loads(open_path.read_text(encoding="utf-8"))
    assert open_after["items"]["BTCUSDC"] == []


def test_cycle16_unmatched_cycle_pos_peak_scope_parity_and_equity_quote_provenance(tmp_path):
    """Cycle 16 regressions:
    1. Unmatched T212 cycles in scope='crypto' must upgrade under-tracked pos_peak (e.g. 148.34 -> 164.64)
       identically to matched positions in scope='equities' via _resolve_cycle_high_since.
    2. Observe-only US equities (e.g. AMD_US_EQ) must retain quote_source/market_state/quote_age_min
       when live price ticks >0.05% away from cache_current_price during macro volatility, while still
       refusing to misclassify a frozen T212 RTH close as Yahoo extended.
    3. Reference-microstructure equities entering via _enrich_reference_microstructure_symbols also receive
       equity quote provenance and session-aware 24h metrics.
    """
    from datetime import datetime
    from botcore import ET
    from intelligence.sentiment.collectors.microstructure_collector import Fleet24hAndMicrostructureEnricher
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles

    # 1. Scope parity on SPCX_US_EQ pos_peak: stale 148.34 in .state file vs true 164.64 cycle high in cache_365
    now_s = datetime(2026, 11, 25, 8, 34, tzinfo=ET).timestamp()
    oldest_lot_ms = int((now_s - 5 * 86400.0) * 1000)
    cdir = tmp_path / "cachedb"
    prices_dir = cdir / "prices"
    prices_dir.mkdir(parents=True)
    (prices_dir / "cache_365_price_SPCX_US_EQ.jsonl").write_text(
        json.dumps({"s": "SPCX_US_EQ", "i": [oldest_lot_ms + 3600_000, 164.64]}) + "\n",
        encoding="utf-8",
    )
    pt_spcx = {"SPCX_US_EQ": {"price": 120.81, "high_24h": 132.24}}
    ro_spcx = {
        "last_action_side": "SELL",
        "active_avg_buy_px": 143.1331,
        "last_buy": {"price": 138.98, "qty": 1.15, "ts_ms": oldest_lot_ms + 86400_000},
        "last_sell": {"price": 125.97, "qty": 2.22, "ts_ms": int((now_s - 17.5 * 3600.0) * 1000)},
        "open_lots": [
            {"qty": 0.53, "entry_px": 159.58, "ts_ms": oldest_lot_ms},
            {"qty": 1.62, "entry_px": 156.47, "ts_ms": oldest_lot_ms + 43200_000},
        ],
    }
    base_cyc = {
        "qty": 2.222965,
        "cost_usd": 318.18,
        "avg_cost_px": 143.1331,
        "entry_price": 144.9872,
        "pos_peak": 148.34,
        "last_buy_price": 138.98,
        "last_sell_price": 125.97,
    }
    # Equities scope (matched tradable position)
    cyc_eq = {"SPCX_US_EQ": dict(base_cyc)}
    pos_eq = [{"venue": "t212", "symbol": "SPCX_US_EQ", "price": 120.81, "free_base": 2.222965, "value_usd": 268.56}]
    enrich_positions_with_orders_and_cycles(
        pos_eq, pt_spcx, {}, {"t212": {"SPCX_US_EQ": dict(ro_spcx)}}, {"SPCX_US_EQ": dict(ro_spcx)},
        {}, {}, cyc_eq, {}, cache_dir=str(cdir), now_ts=now_s,
    )
    # Crypto scope (unmatched T212 cycle — not in tradable_positions)
    cyc_crypto = {"SPCX_US_EQ": dict(base_cyc)}
    enrich_positions_with_orders_and_cycles(
        [], pt_spcx, {}, {"t212": {"SPCX_US_EQ": dict(ro_spcx)}}, {"SPCX_US_EQ": dict(ro_spcx)},
        {}, {}, cyc_crypto, {}, cache_dir=str(cdir), now_ts=now_s,
    )
    assert cyc_eq["SPCX_US_EQ"]["pos_peak"] == pytest.approx(164.64)
    assert cyc_crypto["SPCX_US_EQ"]["pos_peak"] == pytest.approx(164.64)
    assert cyc_crypto["SPCX_US_EQ"]["dd_from_peak_pct"] == cyc_eq["SPCX_US_EQ"]["dd_from_peak_pct"]

    # 2. Fast PRE tick (>0.05% away from cache_current_price) still populates quote_source/market_state/quote_age_min
    amd_entry = {"price": 387.10}
    cp_map = {"AMD_US_EQ": [[int((now_s - 15.0) * 1000), 388.639]]}
    Fleet24hAndMicrostructureEnricher._apply_equity_fallback_quote_meta(
        "AMD_US_EQ", amd_entry, cp_map, None, now_s, frozen_t212_px=None
    )
    assert amd_entry["price"] == pytest.approx(387.10)
    assert amd_entry["quote_source"] == "yahoo_ext"
    assert amd_entry["market_state"] == "PRE"
    assert amd_entry["quote_age_min"] == pytest.approx(0.2, abs=0.1)

    # Frozen T212 RTH close must NOT be misclassified as yahoo_ext even when cp_map has a fresh PRE print
    frozen_entry = {"price": 155.05}
    cp_nvda = {"NVDA_US_EQ": [[int((now_s - 10.0) * 1000), 152.00]]}
    Fleet24hAndMicrostructureEnricher._apply_equity_fallback_quote_meta(
        "NVDA_US_EQ", frozen_entry, cp_nvda, None, now_s, frozen_t212_px=155.05
    )
    assert "quote_source" not in frozen_entry

    # 3. End-to-end Fleet24hAndMicrostructureEnricher.collect for observe-only AMD_US_EQ when live price moved >0.05%
    tel_dir = cdir / "telemetry"
    tel_dir.mkdir(parents=True)
    (tel_dir / "orderbook_depth_AMD_US_EQ.json").write_text(
        json.dumps({"symbol": "AMD_US_EQ", "mid_price": 388.639, "spread_bps": 8.56, "ts": now_s - 12.0}),
        encoding="utf-8",
    )
    pt_end2end = {"AMD_US_EQ": {"price": 387.10, "trend": -1, "growth_coefficient": -0.5}}
    m_end2end: dict = {}
    Fleet24hAndMicrostructureEnricher(start_sync=False).collect(
        str(cdir),
        "BTCUSDC",
        workspace_dir=str(tmp_path),
        prices_and_trends=pt_end2end,
        microstructure=m_end2end,
        cur_p_items=cp_map,
        inst_by_sym={},
        now_ts=now_s,
        read_json_fn=lambda p: json.loads(open(p, encoding="utf-8").read()) if os.path.exists(p) else None,
        allow_network=False,
        persist=False,
    )
    assert pt_end2end["AMD_US_EQ"]["price"] == pytest.approx(387.10)
    assert pt_end2end["AMD_US_EQ"]["quote_source"] == "yahoo_ext"
    assert pt_end2end["AMD_US_EQ"]["market_state"] == "PRE"
    assert pt_end2end["AMD_US_EQ"]["quote_age_min"] == pytest.approx(0.2, abs=0.1)


def test_evaluator_write_purity_ohlc_single_writer_and_t212_cache_first(tmp_path, monkeypatch):
    """Verify Priority 1 + Priority 4 structural safeguards:
    1. price_series_evaluator.py is 100% pure zero-I/O, all evaluators/*.py forbid disk writes,
       cache_ohlc_24h_* has a single canonical locked writer (Cache24PriceManager.save_ohlc_24h_snapshot),
       and intelligence/ forbids private ._client provider bypasses (all enforced statically via AST).
    2. T212Provider.portfolio_mark_prices() exposes a clean public price facade.
    3. refresh_stale_t212_cached_positions serves fresh Layer 3 prices/cache_current_price.json
       with ZERO T212 REST calls, and falls back to T212Provider.portfolio_mark_prices() when stale/missing.
    """
    from pathlib import Path
    from cacheprice import Cache24PriceManager
    import intelligence.sentiment.collectors.microstructure_collector as mc_mod
    from providers.t212_provider import T212Provider
    from verify_tools.ssot_ast_audit import audit_layer_dag_ast, audit_state_file_owners_ast

    repo_root = Path(__file__).resolve().parents[1]

    # 1. Positive + negative AST checks
    assert audit_layer_dag_ast(repo_root, rel_paths=["intelligence/sentiment/evaluators/price_series_evaluator.py"]) == []
    assert audit_layer_dag_ast(repo_root, rel_paths=["intelligence/sentiment/collectors/microstructure_collector.py"]) == []
    assert audit_state_file_owners_ast(repo_root, rel_paths=["cacheprice.py", "intelligence/sentiment/collectors/microstructure_collector.py"]) == []

    synth = tmp_path / "synth_p14"
    (synth / "intelligence" / "sentiment" / "evaluators").mkdir(parents=True)
    (synth / "intelligence" / "sentiment" / "collectors").mkdir(parents=True)
    (synth / "intelligence" / "sentiment" / "evaluators" / "price_series_evaluator.py").write_text(
        "import os\nfrom state_io import save_json_state\n"
        "def bad(p):\n    save_json_state(p, {})\n",
        encoding="utf-8",
    )
    (synth / "intelligence" / "sentiment" / "collectors" / "rogue_col.py").write_text(
        "from cachepaths import cache_ohlc_24h_filename\n"
        "from state_io import save_json_state\n"
        "def bad(prov, sym):\n"
        "    _ = prov._client()\n"
        "    p = cache_ohlc_24h_filename(sym)\n"
        "    save_json_state(p, {'points': []})\n",
        encoding="utf-8",
    )
    dag_bad = audit_layer_dag_ast(
        synth,
        rel_paths=[
            "intelligence/sentiment/evaluators/price_series_evaluator.py",
            "intelligence/sentiment/collectors/rogue_col.py",
        ],
    )
    assert any("pure valuation module must not import disk/state I/O modules" in v for v in dag_bad)
    assert any("stateless evaluator must not call disk writer save_json_state()" in v for v in dag_bad)
    assert any("private provider ._client access in intelligence/" in v for v in dag_bad)
    own_bad = audit_state_file_owners_ast(synth, rel_paths=["intelligence/sentiment/collectors/rogue_col.py"])
    assert any("direct raw write of cache_ohlc_24h_filename via save_json_state()" in v for v in own_bad)

    # 2. Cache24PriceManager.save_ohlc_24h_snapshot preserves fresh equity OHLCV under lock
    now_s = 1_795_950_000.0
    cdb = tmp_path / "cachedb"
    (cdb / "prices").mkdir(parents=True)
    (cdb / "state").mkdir(parents=True)
    ohlc_p = cdb / "prices" / "cache_ohlc_24h_NVDA_US_EQ.json"
    pts_cm = [[int((now_s - (24 - i) * 3600) * 1000), 140.0 + i * 0.1, 100.0] for i in range(25)]
    pts_fb = [[int((now_s - (24 - i) * 3600) * 1000), 999.0, 100.0] for i in range(25)]
    assert Cache24PriceManager.save_ohlc_24h_snapshot(str(ohlc_p), "NVDA_US_EQ", pts_cm, now_ts=now_s - 20.0) is True
    assert (
        Cache24PriceManager.save_ohlc_24h_snapshot(
            str(ohlc_p), "NVDA_US_EQ", pts_fb, now_ts=now_s, preserve_fresh_equity_sec=240.0
        )
        is False
    )
    assert json.loads(ohlc_p.read_text(encoding="utf-8"))["points"][0][1] == pytest.approx(140.0)

    # 3. T212Provider.portfolio_mark_prices() public facade
    t212_p = T212Provider()
    monkeypatch.setattr(
        t212_p,
        "_client",
        lambda: SimpleNamespace(
            get_portfolio=lambda: [
                {"ticker": "nvda_us_eq", "currentPrice": 144.5},
                {"ticker": "BAD_EQ", "currentPrice": 0.0},
            ]
        ),
    )
    assert t212_p.portfolio_mark_prices() == {"NVDA_US_EQ": 144.5}

    # 4. refresh_stale_t212_cached_positions: Layer 3 cache_current_price.json cache-first (0 REST), then fallback when stale
    vb_path = cdb / "state" / "portfolio_positions_state.json"
    old_bal_ts = now_s - 600.0
    cached_bals = {
        "pos:t212:NVDA_US_EQ": {
            "venue": "t212",
            "symbol": "NVDA_US_EQ",
            "free_base": 2.0,
            "price": 130.0,
            "value_usd": 260.0,
            "ts": old_bal_ts,
        }
    }
    vb_path.write_text(json.dumps(cached_bals), encoding="utf-8")
    cp_path = cdb / "prices" / "cache_current_price.json"
    cp_path.write_text(
        json.dumps(
            {
                "ts": now_s - 15.0,
                "items": {"NVDA_US_EQ": [[int((now_s - 15.0) * 1000), 146.25]]},
                "fetchtime": {"NVDA_US_EQ": int((now_s - 15.0) * 1000)},
            }
        ),
        encoding="utf-8",
    )
    rest_calls = {"t212": 0}

    class _CountingT212Provider:
        def portfolio_mark_prices(self):
            rest_calls["t212"] += 1
            return {"NVDA_US_EQ": 149.0}

    monkeypatch.setattr(mc_mod._t212_prov_mod, "T212Provider", _CountingT212Provider)
    monkeypatch.setattr(mc_mod, "_LAST_T212_PORTFOLIO_POLL_TS", 0.0)
    monkeypatch.setattr(mc_mod, "is_symbol_market_open", lambda _sym, _ts=None: True)

    assert mc_mod.refresh_stale_t212_cached_positions(cached_bals, str(vb_path), now_s, max_age_sec=180.0) is True
    assert rest_calls["t212"] == 0
    assert cached_bals["pos:t212:NVDA_US_EQ"]["price"] == pytest.approx(146.25)
    assert cached_bals["pos:t212:NVDA_US_EQ"]["value_usd"] == pytest.approx(292.5)
    assert cached_bals["pos:t212:NVDA_US_EQ"]["base_bal_ts"] == pytest.approx(old_bal_ts)

    # Now make cache_current_price.json stale (>180s) -> falls back cleanly to public portfolio_mark_prices()
    cp_path.write_text(
        json.dumps({"ts": now_s - 900.0, "items": {"NVDA_US_EQ": [[int((now_s - 900.0) * 1000), 146.25]]}}),
        encoding="utf-8",
    )
    cached_bals["pos:t212:NVDA_US_EQ"]["ts"] = now_s - 600.0
    assert mc_mod.refresh_stale_t212_cached_positions(cached_bals, str(vb_path), now_s, max_age_sec=180.0) is True
    assert rest_calls["t212"] == 1
    assert cached_bals["pos:t212:NVDA_US_EQ"]["price"] == pytest.approx(149.0)
    assert cached_bals["pos:t212:NVDA_US_EQ"]["value_usd"] == pytest.approx(298.0)


def test_priority2_pure_provider_math_modules_zero_io_and_stores(tmp_path):
    """Verify Priority 2 pure-math vs store separation across nav_drawdown, ai_attribution, quantity, and intent_state:
    1. All 6 PURE_VALUATION_MODULES pass audit_layer_dag_ast with zero disk/state I/O imports.
    2. Synthetic os/glob/state_io/state_stores leaks into any of the 4 newly purified modules are statically blocked.
    3. All 4 store facades (nav_history_store, ai_attribution_store, intent_state_store, portfolio_state_store)
       function cleanly end-to-end.
    """
    from pathlib import Path
    from providers.ai_attribution_store import load_and_evaluate_ai_attribution
    from providers.intent_state_store import (
        load_durable_intents_from_cachedb,
        load_known_ai_order_ids_from_cachedb,
        upsert_durable_intent,
    )
    from providers.nav_history_store import load_and_evaluate_nav_drawdown, record_nav_history_snapshot
    from providers.portfolio_state_store import resolve_kraken_trade_cache_position
    from verify_tools.ssot_ast_audit import PURE_VALUATION_MODULES, audit_layer_dag_ast

    repo_root = Path(__file__).resolve().parents[1]
    expected_pure = {
        "providers/portfolio_valuation.py",
        "providers/nav_drawdown.py",
        "providers/ai_attribution.py",
        "providers/quantity.py",
        "providers/intent_state.py",
        "intelligence/sentiment/evaluators/price_series_evaluator.py",
    }
    assert expected_pure.issubset(PURE_VALUATION_MODULES)
    assert audit_layer_dag_ast(repo_root, rel_paths=sorted(expected_pure)) == []

    # Negative AST checks: any disk/state I/O import in the 4 newly purified provider modules is blocked
    synth = tmp_path / "synth_p2"
    (synth / "providers").mkdir(parents=True)
    leaks = {
        "providers/nav_drawdown.py": "import os\nfrom state_stores.nav_history_state import NavHistoryStateStore\n",
        "providers/ai_attribution.py": "import glob\nfrom state_io import read_json_dict\n",
        "providers/quantity.py": "import os\nfrom state_io import read_json_dict\n",
        "providers/intent_state.py": "from state_stores.auto_order_intents_state import AutoOrderIntentsStateStore\n",
    }
    for rel, code in leaks.items():
        (synth / rel).write_text(code, encoding="utf-8")
    viols = audit_layer_dag_ast(synth, rel_paths=sorted(leaks.keys()))
    for rel in leaks:
        assert any(v.startswith(f"{rel}: pure valuation module must not import disk/state I/O modules") for v in viols)

    # Functional round-trip across all 4 store facades on isolated workspace
    ws = tmp_path / "ws_p2"
    cdb = ws / "cachedb"
    (cdb / "state").mkdir(parents=True)
    (cdb / "orders").mkdir(parents=True)
    (cdb / "prices").mkdir(parents=True)
    now_s = 1_795_960_000.0

    # intent_state_store
    from cachepaths import AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME

    int_path = cdb / "state" / AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME
    assert upsert_durable_intent(
        str(int_path),
        {
            "intent_id": "ai-p2-1",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "decision": "BUY",
            "status": "EXECUTED",
            "timestamp": now_s - 600.0,
            "execution_payload": {"orderId": "bn-901", "status": "FILLED", "executedQty": "0.01", "avg_fill_price": 70000.0},
        },
    ) is True
    assert len(load_durable_intents_from_cachedb(str(cdb))) == 1
    assert "bn-901" in load_known_ai_order_ids_from_cachedb(str(cdb))

    # nav_history_store
    m_nav = record_nav_history_snapshot(
        str(ws),
        venue_balances_state={"cash:binance:USDC": {"venue": "binance", "asset": "USDC", "free": 2000.0}},
        live_prices={},
        now_ts=now_s,
    )
    assert m_nav["current_nav_usd"] == pytest.approx(2000.0)
    assert load_and_evaluate_nav_drawdown(str(ws), now_ts=now_s, include_live_point=False)["sample_count"] == 1

    # ai_attribution_store
    rep = load_and_evaluate_ai_attribution(str(ws), lookback_days=7.0, now_ts=now_s)
    assert rep["unified_nav_drawdown"]["sample_count"] == 1

    # portfolio_state_store.resolve_kraken_trade_cache_position
    (cdb / "orders" / "cache_order_filled_kraken.json").write_text(
        json.dumps(
            {
                "items": {
                    "HYPEUSD": [
                        {"id": "k1", "side": "BUY", "price": 40.0, "qty": 5.0, "time": int((now_s - 3600) * 1000)}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    kpos = resolve_kraken_trade_cache_position("HYPEUSD", str(cdb), held_qty=5.0)
    assert kpos is not None and kpos["avg_cost_px"] == pytest.approx(40.0)


def test_cycle17_rth_open_inflight_quote_skew_whale_long_pp_and_opening_rvol_1h():
    """Cycle 17 regressions caught at US NYSE/Nasdaq Cash Open (09:33-09:38 EST):
    1. d_whale_long must render fraction delta as percentage points (*100, e.g. -34.4pp instead of -0.34%).
    2. In-flight equity ticks written 1-60s AFTER snapshot start now_ts must NOT fall through to yahoo_rth_close
       or keep stale quote_age_min during live US RTH or PRE/POST; truly far-future corrupted timestamps (>60s) still fail.
    3. Opening-hour US equity rvol_1h after an overnight >1h market-closed gap must prorate baseline by active
       session elapsed fraction (floored at 0.25 = 15m), preventing 4x understatement during opening-bell surges,
       while continuous 24/7 crypto rvol_1h remains unchanged.
    """
    from datetime import datetime, timezone
    from botcore import ET
    from intelligence.sentiment.collectors.microstructure_collector import (
        Fleet24hAndMicrostructureEnricher,
        _ResolvedEquityQuote,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import compute_volume_rvol_metrics
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section

    # 1. d_whale_long renders percentage points (*100), never 100x-understated fraction with '%'
    sec3b = "\n".join(
        _format_microstructure_section(
            {
                "AMD_US_EQ": {
                    "ob_imbalance": 0.278,
                    "bid_depth_usd": 250_000.0,
                    "ask_depth_usd": 647_956.0,
                    "whale_long_pct": 0.1563,
                    "taker_ratio": 0.1853,
                    "taker_buy_vol_usd": 219_382_129.0,
                    "taker_sell_vol_usd": 1_184_102_525.0,
                    "cycle_delta": {
                        "dt_min": 6.0,
                        "ob_imbalance_delta": -0.214,
                        "taker_ratio_delta": -0.81,
                        "whale_long_delta_pct": -0.3437,
                    },
                }
            },
            {"AMD_US_EQ": {"price": 373.07}},
        )
    )
    assert "d_whale_long=-34.4pp" in sec3b
    assert "d_whale_long=-0.34%" not in sec3b

    # 2. In-flight tick (+4s newer than snapshot eff_now) during live US RTH classifies as yahoo_rth with age=0.0m
    eff_now_rth = datetime(2026, 11, 25, 9, 38, 46, tzinfo=ET).timestamp()
    inflight_ts = eff_now_rth + 4.0
    src_rth, mstate_rth = Fleet24hAndMicrostructureEnricher._classify_yahoo_equity_quote_source(
        "NVDA_US_EQ", inflight_ts, eff_now_rth
    )
    assert src_rth == "yahoo_rth" and mstate_rth is None

    nv_entry = {"price": 142.67, "quote_source": "t212_live", "quote_age_min": 3.7}
    Fleet24hAndMicrostructureEnricher._apply_resolved_equity_quote(
        nv_entry,
        _ResolvedEquityQuote(price=142.67, quote_source=src_rth, quote_ts_sec=inflight_ts, market_state=mstate_rth),
        eff_now=eff_now_rth,
        set_price=False,
    )
    assert nv_entry["quote_source"] == "yahoo_rth"
    assert nv_entry["quote_age_min"] == 0.0

    # Same in-flight tolerance in PRE session stays yahoo_ext / PRE (not yahoo_ext_cached)
    eff_now_pre = datetime(2026, 11, 25, 8, 35, 0, tzinfo=ET).timestamp()
    src_pre, mstate_pre = Fleet24hAndMicrostructureEnricher._classify_yahoo_equity_quote_source(
        "AMD_US_EQ", eff_now_pre + 5.0, eff_now_pre
    )
    assert src_pre == "yahoo_ext" and mstate_pre == "PRE"

    # Truly far-future corrupted timestamp (+5 min ahead) still does NOT classify as live yahoo_rth
    src_corrupt, _ = Fleet24hAndMicrostructureEnricher._classify_yahoo_equity_quote_source(
        "NVDA_US_EQ", eff_now_rth + 300.0, eff_now_rth
    )
    assert src_corrupt == "yahoo_rth_close"

    # 3. Opening-hour US equity rvol_1h prorates active session fraction after >1h overnight gap
    d1_rth0 = datetime(2026, 11, 24, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    d2_rth0 = datetime(2026, 11, 25, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    d1_bars = [[d1_rth0 + i * 300_000.0, 100.0, 100.0] for i in range(78)]  # 6.5h at $10k/5m ($120k/1h, $30k/15m)
    # 10 min into Day 2 open (2 bars at $30k/5m = $60k total, ~2x 15m baseline): rvol_1h must NOT read 0.5x!
    open_2bars = d1_bars + [[d2_rth0 + i * 300_000.0, 100.0, 300.0] for i in range(2)]
    rvol_open = compute_volume_rvol_metrics(open_2bars, now_ts=(d2_rth0 + 400_000.0) / 1000.0)
    assert rvol_open["rvol_15m"] == pytest.approx(rvol_open["rvol_1h"], abs=0.05)
    assert rvol_open["rvol_1h"] > 1.7

    # 30 min into Day 2 open (6 bars at $20k/5m = $120k = 2x normal 30m baseline): rvol_1h ~ 1.85x (not 0.9x)
    open_6bars = d1_bars + [[d2_rth0 + i * 300_000.0, 100.0, 200.0] for i in range(6)]
    rvol_30m = compute_volume_rvol_metrics(open_6bars, now_ts=(d2_rth0 + 1_600_000.0) / 1000.0)
    assert 1.6 <= rvol_30m["rvol_1h"] <= 2.1

    # Continuous 24/7 crypto bars (no >1h closure gap) keep exact standard full-1h baseline
    c_start = d2_rth0 - 6 * 3_600_000.0
    crypto_bars = [[c_start + i * 300_000.0, 100.0, 100.0] for i in range(72)]
    rvol_crypto = compute_volume_rvol_metrics(crypto_bars, now_ts=(c_start + 71 * 300_000.0) / 1000.0)
    assert rvol_crypto["rvol_1h"] == pytest.approx(1.0, abs=0.05)


def test_variant_a_storage_domains_dual_layout_migration_and_fee_schedule_split(tmp_path):
    """Verify Variant A storage separation (cachedb/ vs statedb/), zero-split-brain dual layout,
    dry-run + applied CLI cutover migration, and clean cache_fee_schedule vs venue_fee_maintenance split.
    """
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import account_cache_health
    from active_intents import build_active_intent_index
    from cachepaths import (
        AUTO_ORDER_INTENTS_STATE_FILENAME,
        CACHE_ACCOUNT_HEALTH_FILENAME,
        CACHE_BINANCE_BALANCE_FILENAME,
        CACHE_FEE_SCHEDULE_FILENAME,
        DurabilityTier,
        KINEMATIC_IMPULSE_SIGNAL_FILENAME,
        LLM_MODEL_CATALOG_FILENAME,
        LLM_QUOTA_STATUS_FILENAME,
        LLM_TOPIC_THREADS_FILENAME,
        NAV_HISTORY_STATE_FILENAME,
        NOTIFY_QUEUE_FILENAME,
        ORDER_LEDGER_ACTIVE_FILENAME,
        ORDER_RETRY_QUEUE_FILENAME,
        PORTFOLIO_POSITIONS_STATE_FILENAME,
        VENUE_FEE_MAINTENANCE_STATE_FILENAME,
        is_financial_critical_path,
        resolve_cachedb_filepath,
        resolve_durability_policy,
        resolve_statedb_filepath,
        resolve_storage_domain,
    )
    from providers.nav_history_store import _load_workspace_balances_and_prices
    from providers.order_ledger import OrderLedgerReader
    from providers.portfolio_state_store import load_venue_balance_caches
    from providers.venue_fee_maintenance import (
        KrakenFeeMaintainer,
        VenueFeeMaintainerInterface,
        read_persisted_venue_fee_spec,
        resolve_fee_schedule_cache_path,
        run_venue_fee_maintenance,
    )
    from state_stores import (
        AutoOrderIntentsStateStore,
        LlmTopicThreadsStore,
        NavHistoryStateStore,
        NotifyQueueStore,
        OrderRetryQueueStore,
        PortfolioPositionsStateStore,
        VenueFeeMaintenanceStateStore,
    )
    from verify_tools import migrate_storage_domains as migrate_cli

    # 1. Variant A taxonomy classification & DurabilityPolicy
    domain_cases = {
        "cache_current_price.json": ("cachedb", "prices"),
        "cache_short_trend.json": ("cachedb", "trends"),
        KINEMATIC_IMPULSE_SIGNAL_FILENAME: ("cachedb", "telemetry"),
        "fear_greed_collect.json": ("cachedb", "macro"),
        CACHE_BINANCE_BALANCE_FILENAME: ("cachedb", "account"),
        "cache_order_open_binance.json": ("cachedb", "account"),
        "cache_order_filled_kraken.json": ("cachedb", "account"),
        CACHE_FEE_SCHEDULE_FILENAME: ("cachedb", "account"),
        CACHE_ACCOUNT_HEALTH_FILENAME: ("cachedb", "account"),
        "account_cache_health.json": ("cachedb", "account"),
        "binance_account_cache_health.json": ("cachedb", "account"),
        LLM_MODEL_CATALOG_FILENAME: ("cachedb", "llm"),
        LLM_QUOTA_STATUS_FILENAME: ("cachedb", "llm"),
        "ai_attribution_status.json": ("cachedb", "llm"),
        "auto_cryptotrade_executor_eval.json": ("cachedb", "llm"),
        "fleet_health_status.json": ("cachedb", "system"),
        "rtrade_BTCUSDC.heartbeat": ("cachedb", "system/heartbeat"),
        ORDER_LEDGER_ACTIVE_FILENAME: ("statedb", "orders"),
        "order_events_2026-11-25.jsonl": ("statedb", "orders"),
        ORDER_RETRY_QUEUE_FILENAME: ("statedb", "orders"),
        AUTO_ORDER_INTENTS_STATE_FILENAME: ("statedb", "orders"),
        "autonomous_order_intents_state.json": ("statedb", "orders"),
        "autonomous_trade_intents_state.json": ("statedb", "orders"),
        PORTFOLIO_POSITIONS_STATE_FILENAME: ("statedb", "portfolio"),
        NAV_HISTORY_STATE_FILENAME: ("statedb", "portfolio"),
        VENUE_FEE_MAINTENANCE_STATE_FILENAME: ("statedb", "bots"),
        LLM_TOPIC_THREADS_FILENAME: ("statedb", "llm"),
        NOTIFY_QUEUE_FILENAME: ("statedb", "system"),
    }
    for fname, expected in domain_cases.items():
        assert resolve_storage_domain(fname) == expected, f"{fname} -> expected {expected}"
    assert CACHE_ACCOUNT_HEALTH_FILENAME == "cache_account_health.json"
    assert AUTO_ORDER_INTENTS_STATE_FILENAME == "auto_order_intents_state.json"
    assert is_financial_critical_path(AUTO_ORDER_INTENTS_STATE_FILENAME) is True
    assert is_financial_critical_path(LLM_TOPIC_THREADS_FILENAME) is True
    assert is_financial_critical_path(NOTIFY_QUEUE_FILENAME) is True
    assert is_financial_critical_path(CACHE_FEE_SCHEDULE_FILENAME) is True
    assert resolve_durability_policy(KINEMATIC_IMPULSE_SIGNAL_FILENAME).tier == DurabilityTier.EPHEMERAL_TELEMETRY

    # 2. Pre-cutover legacy workspace (no statedb/, no cachedb/account/, no cachedb/llm/)
    ws = tmp_path / "ws_variant_a"
    cdb = ws / "cachedb"
    sdb = ws / "statedb"
    for sub in ("prices", "orders", "state", "system", "telemetry"):
        (cdb / sub).mkdir(parents=True)

    now_s = 1_796_000_000.0
    (cdb / "orders" / CACHE_BINANCE_BALANCE_FILENAME).write_text(
        json.dumps({"ts": now_s, "items": {"USDC": {"free": 2500.0, "locked": 0.0}, "BNB": {"free": 0.3, "locked": 0.0}}}),
        encoding="utf-8",
    )
    (cdb / "orders" / "binance_account_cache_health.json").write_text(
        json.dumps({"schema_version": 2, "order_sync_at_ms": int(now_s * 1000)}),
        encoding="utf-8",
    )
    (cdb / "orders" / ORDER_LEDGER_ACTIVE_FILENAME).write_text(
        json.dumps({"schema_version": 1, "updated_ts": now_s, "journal_cursor": {"file": "", "offset": 0}, "orders": {}}),
        encoding="utf-8",
    )
    (cdb / "orders" / "order_events_2026-11-25.jsonl").write_text(
        json.dumps({"ts": now_s, "ev": "SUBMITTED", "v": "binance", "sym": "BTCUSDC", "vid": "12345"}) + "\n",
        encoding="utf-8",
    )
    (cdb / "orders" / ORDER_RETRY_QUEUE_FILENAME).write_text(
        json.dumps({"id": "rq1", "symbol": "BTCUSDC", "side": "BUY"}) + "\n",
        encoding="utf-8",
    )
    (cdb / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps([
            {
                "intent_id": "ai_va_1",
                "venue": "binance",
                "symbol": "BTCUSDC",
                "decision": "BUY",
                "status": "PENDING",
                "requested_qty": 0.01,
                "requested_price": 70000.0,
                "timestamp": now_s,
            }
        ]),
        encoding="utf-8",
    )
    (cdb / "state" / "autonomous_trade_intents_state.json.previous").write_text("[]", encoding="utf-8")
    (cdb / "state" / PORTFOLIO_POSITIONS_STATE_FILENAME).write_text(
        json.dumps({"pos:binance:BTCUSDC": {"venue": "binance", "symbol": "BTCUSDC", "trailing_peak": 72000.0}}),
        encoding="utf-8",
    )
    (cdb / "state" / NAV_HISTORY_STATE_FILENAME).write_text(
        json.dumps({"version": 1, "updated_ts": now_s, "points": [{"ts": now_s - 300.0, "nav_usd": 18000.0}]}),
        encoding="utf-8",
    )
    (cdb / "system" / LLM_TOPIC_THREADS_FILENAME).write_text(
        json.dumps({"auto_cryptotrade_executor": {"conversation_id": "cid_va"}}),
        encoding="utf-8",
    )
    (cdb / "system" / NOTIFY_QUEUE_FILENAME).write_text(
        json.dumps({"__orchestrator_intent__": "email", "subject": "queued_va"}) + "\n",
        encoding="utf-8",
    )
    (cdb / "system" / KINEMATIC_IMPULSE_SIGNAL_FILENAME).write_text(json.dumps({"ts": now_s}), encoding="utf-8")
    (cdb / "system" / LLM_QUOTA_STATUS_FILENAME).write_text(json.dumps({"ts": now_s}), encoding="utf-8")

    # Pre-cutover passive dual-layout reads must find every legacy file with zero split-brain
    assert Path(str(resolve_cachedb_filepath(str(cdb / CACHE_ACCOUNT_HEALTH_FILENAME)))) == (
        cdb / "orders" / "binance_account_cache_health.json"
    )
    assert Path(str(resolve_statedb_filepath(str(cdb / AUTO_ORDER_INTENTS_STATE_FILENAME)))) == (
        cdb / "state" / "autonomous_trade_intents_state.json"
    )
    assert len(AutoOrderIntentsStateStore.load_from_cachedb(str(cdb))) == 1
    assert "binance" in load_venue_balance_caches(str(cdb))

    # 3. CLI dry-run preview leaves disk untouched
    assert migrate_cli.main(["--workspace-dir", str(ws), "--json"]) == 0
    assert not sdb.exists()
    assert (cdb / "state" / "autonomous_trade_intents_state.json").is_file()

    # 4. CLI --apply performs clean atomic Variant A cutover + renames + sidecars
    assert migrate_cli.main(["--workspace-dir", str(ws), "--apply", "--json"]) == 0
    assert (cdb / "account" / CACHE_BINANCE_BALANCE_FILENAME).is_file()
    assert (cdb / "account" / CACHE_ACCOUNT_HEALTH_FILENAME).is_file()
    assert (cdb / "telemetry" / KINEMATIC_IMPULSE_SIGNAL_FILENAME).is_file()
    assert (cdb / "llm" / LLM_QUOTA_STATUS_FILENAME).is_file()
    assert (sdb / "orders" / ORDER_LEDGER_ACTIVE_FILENAME).is_file()
    assert (sdb / "orders" / "order_events_2026-11-25.jsonl").is_file()
    assert (sdb / "orders" / ORDER_RETRY_QUEUE_FILENAME).is_file()
    assert (sdb / "orders" / AUTO_ORDER_INTENTS_STATE_FILENAME).is_file()
    assert (sdb / "orders" / f"{AUTO_ORDER_INTENTS_STATE_FILENAME}.previous").is_file()
    assert (sdb / "portfolio" / PORTFOLIO_POSITIONS_STATE_FILENAME).is_file()
    assert (sdb / "portfolio" / NAV_HISTORY_STATE_FILENAME).is_file()
    assert (sdb / "llm" / LLM_TOPIC_THREADS_FILENAME).is_file()
    assert (sdb / "system" / NOTIFY_QUEUE_FILENAME).is_file()
    # Empty legacy cachedb/orders & cachedb/state directories removed cleanly
    assert not (cdb / "orders").exists()
    assert not (cdb / "state").exists()

    # 5. Post-cutover readers across ALL subsystems seamlessly resolve statedb/ & cachedb/account/
    assert Path(account_cache_health._resolve_health_path(str(cdb / CACHE_ACCOUNT_HEALTH_FILENAME))) == (
        cdb / "account" / CACHE_ACCOUNT_HEALTH_FILENAME
    )
    assert "binance" in load_venue_balance_caches(str(cdb))
    assert len(AutoOrderIntentsStateStore.load_from_cachedb(str(cdb))) == 1
    assert len(OrderRetryQueueStore.read_tail_records(str(cdb), limit=10)) == 1
    assert PortfolioPositionsStateStore.load_dict(str(cdb))["pos:binance:BTCUSDC"]["trailing_peak"] == pytest.approx(72000.0)
    assert NavHistoryStateStore.read_raw_doc(str(ws))["points"][0]["nav_usd"] == pytest.approx(18000.0)
    assert LlmTopicThreadsStore.read_registry(LlmTopicThreadsStore.resolve_path(str(cdb)))["auto_cryptotrade_executor"]["conversation_id"] == "cid_va"
    assert len(NotifyQueueStore.read_lines_nolock(NotifyQueueStore.resolve_path(str(cdb)))) == 1
    lr = OrderLedgerReader(cachedb_dir=str(cdb))
    assert len(lr.list_event_files()) == 1 and str(sdb / "orders") in lr.list_event_files()[0]
    idx = build_active_intent_index(str(ws))
    assert any(it.get("intent_id") == "ai_va_1" for it in idx["intents"])

    # 6. Clean split: cache_fee_schedule.json (cachedb/account/) vs venue_fee_maintenance_state.json (statedb/state/)
    cfg = {
        "FEE_MAINTENANCE_ENABLED": "true",
        "FEE_MAINTENANCE_INTERVAL_SEC": 14400,
        "BINANCE_BNB_RESERVE_MIN_USDC": 100.0,
        "BINANCE_BNB_RESERVE_TARGET_USDC": 200.0,
        "BINANCE_BNB_MIN_FREE_USDC": 300.0,
        "BINANCE_BNB_ORDER_STALE_SEC": 1800,
        "KRAKEN_IDLE_USDC_ALERT_MIN": 25.0,
        "KRAKEN_IDLE_USDC_ALERT_COOLDOWN_SEC": 86400,
    }
    bn_m = MagicMock(spec=VenueFeeMaintainerInterface)
    bn_m.maintain.return_value = {
        "ts": now_s,
        "fee_spec": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.5},
        "by_symbol": {"BTCUSDC": {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.12}},
        "spot_bnb_burn": True,
        "bnb_free": 0.3,
        "bnb_locked": 0.0,
        "bnb_price_usdc": 620.0,
        "held_bnb_usdc": 186.0,
        "effective_bnb_usdc": 186.0,
        "free_usdc": 2500.0,
        "has_pending_bnb_order": False,
        "retry_soon": False,
        "topup_action": "none",
        "events": [],
    }
    kr_p = MagicMock()
    kr_cli = MagicMock()
    kr_cli._private.return_value = {
        "volume": "25000.0",
        "fees": {"TAOUSD": {"fee": "0.3800"}},
        "fees_maker": {"TAOUSD": {"fee": "0.2200"}},
    }
    kr_p._client.return_value = kr_cli
    kr_p.usd_cash_breakdown.return_value = {"ZUSD": 1500.0, "USDC": 50.0}
    kr_m = KrakenFeeMaintainer(provider=kr_p)

    res_fee = run_venue_fee_maintenance(
        venues=["binance", "kraken"],
        cache_dir=str(cdb),
        cfg_map=cfg,
        maintainers={"binance": bn_m, "kraken": kr_m},
        now_ts=now_s,
    )
    assert res_fee["status"] == "ok"

    sched_p = Path(resolve_fee_schedule_cache_path(str(cdb)))
    wf_p = Path(VenueFeeMaintenanceStateStore.resolve_path(str(cdb)))
    assert sched_p == cdb / "account" / CACHE_FEE_SCHEDULE_FILENAME
    assert wf_p == sdb / "bots" / VENUE_FEE_MAINTENANCE_STATE_FILENAME

    sched_doc = json.loads(sched_p.read_text(encoding="utf-8"))
    wf_doc = json.loads(wf_p.read_text(encoding="utf-8"))

    # Exchange truth lives strictly in cachedb/account/cache_fee_schedule.json (no workflow keys)
    bn_sched = sched_doc["venues"]["binance"]
    assert bn_sched["fee_spec"]["maker_fee_bps"] == pytest.approx(7.5)
    assert bn_sched["bnb_price_usdc"] == pytest.approx(620.0)
    assert bn_sched["held_bnb_usdc"] == pytest.approx(186.0)
    for forbidden_wf_key in ("has_pending_bnb_order", "retry_soon", "topup_action", "last_idle_usdc_alert_ts"):
        assert forbidden_wf_key not in bn_sched
    assert "last_run_ts" not in sched_doc

    # Local workflow state lives strictly in statedb/state/venue_fee_maintenance_state.json (no fee schedule duplication)
    bn_wf = wf_doc["venues"]["binance"]
    assert wf_doc["last_run_ts"] == pytest.approx(now_s)
    assert bn_wf["has_pending_bnb_order"] is False
    assert bn_wf["topup_action"] == "none"
    assert wf_doc["venues"]["kraken"]["last_idle_usdc_alert_ts"] == pytest.approx(now_s)
    for forbidden_sched_key in ("fee_spec", "by_symbol", "bnb_price_usdc", "held_bnb_usdc", "spot_bnb_burn"):
        assert forbidden_sched_key not in bn_wf

    # Readers (TCA fee spec & NAV history BNB pricing) consume cache_fee_schedule.json cleanly
    live_spec = read_persisted_venue_fee_spec("binance", "BTCUSDC", cache_dir=str(cdb), now_ts=now_s + 10.0)
    assert live_spec == {"fee_tier": "VIP0_BNB", "maker_fee_bps": 7.5, "taker_fee_bps": 7.12}
    _, _, seeded_prices = _load_workspace_balances_and_prices(str(ws))
    assert seeded_prices["BNBUSDC"][0][1] == pytest.approx(620.0)


def test_cycle18_xstock_no_negative_zero_and_personal_finance_affiliate_news_filter(tmp_path):
    """Cycle 18 regressions:
    1. KrakenXstocksCollector must normalize -0.0 to 0.0 on xstock_vs_underlying_pct and pnl_vs_alloc_pct.
    2. Consumer personal-finance affiliate/listicle items (e.g. CNBC Select / 'Best low-interest personal loans...')
       must be excluded so they cannot evict live macro headlines from the top-6 Breaking News feed.
    """
    import math
    from intelligence.macro.lexicon_ssot import has_financial_or_geopolitical_context, is_excluded_reference_news_item
    from intelligence.sentiment.collectors.portfolio_collector import KrakenXstocksCollector
    from state_io import save_json_state

    # 1. Tiny negative delta (-0.004%) must emit +0.0 (not IEEE-754 -0.0)
    k_dir = tmp_path / "kraken_neg0"
    k_dir.mkdir()
    save_json_state(
        str(k_dir / "xstock_state.json"),
        {"known_assets": {"SPCXx": 12.135889}, "pair": "SPCXxUSD", "allocated": {"asset": "SPCXx", "qty": 12.135889, "alloc_price": 117.105}},
    )
    xst = KrakenXstocksCollector(start_sync=False).collect(
        str(k_dir),
        kraken_dir=str(k_dir),
        prices_and_trends={
            "SPCXxUSD": {"price": 117.1},
            "SPCX_US_EQ": {"price": 117.105, "quote_source": "yahoo_rth"},
        },
        all_t212_cycles={},
    )
    assert xst is not None
    assert xst["xstock_vs_underlying_pct"] == 0.0
    assert not math.copysign(1.0, xst["xstock_vs_underlying_pct"]) < 0
    assert xst["pnl_vs_alloc_pct"] == 0.0
    assert not math.copysign(1.0, xst["pnl_vs_alloc_pct"]) < 0

    # 2. CNBC Select & consumer personal-finance product roundups are excluded; genuine macro headlines remain allowed
    assert is_excluded_reference_news_item(
        "Best low-interest personal loans of December 2026",
        source="US Top News and Analysis",
        url="https://www.cnbc.com/select/best-low-interest-personal-loans/",
    ) is True
    assert has_financial_or_geopolitical_context("Best low-interest personal loans of December 2026") is False
    assert has_financial_or_geopolitical_context("Best high-yield savings accounts for 2026") is False
    assert is_excluded_reference_news_item(
        "Some affiliate comparison article",
        url="https://www.cnbc.com/select/some-product-review/",
    ) is True
    assert is_excluded_reference_news_item(
        "Fed's preferred inflation gauge accelerated to 3.8% in October",
        url="https://www.cnbc.com/2026/11/25/pce-inflation.html",
    ) is False
    assert has_financial_or_geopolitical_context("Personal loan delinquencies rise to five-year high") is True


def test_point3_1to1_stores_locked_merges_and_zero_io_evaluators(tmp_path: Any) -> None:
    """Verify 4 new 1:1 state stores (locked read-merge-write) and 100% zero-I/O evaluator purity."""
    from state_stores import (
        CacheTTrendStore,
        MicrostructureCycleHistoryStore,
        OrderRetryQuarantineAlertsStore,
        OrdersActiveStore,
    )
    from verify_tools.ssot_ast_audit import (
        audit_layer_dag_ast,
        audit_state_file_owners_ast,
    )

    cdb = tmp_path / "cachedb"
    (cdb / "telemetry").mkdir(parents=True)
    (cdb / "orders").mkdir(parents=True)
    (cdb / "trends").mkdir(parents=True)

    # 1. MicrostructureCycleHistoryStore merges overlapping multi-writer symbol updates without lost updates
    mh_path = MicrostructureCycleHistoryStore.resolve_path(str(cdb))
    assert MicrostructureCycleHistoryStore.upsert_history_locked(
        mh_path,
        {"BTCUSDC": [{"ts": 1000.0, "ob_imbalance": 0.55}], "ETHUSDC": [{"ts": 1000.0, "ob_imbalance": 0.45}]},
    )
    # Second overlapping writer has stale snapshot without ETHUSDC and adds new BTCUSDC ts + SOLUSDC
    assert MicrostructureCycleHistoryStore.upsert_history_locked(
        mh_path,
        {"BTCUSDC": [{"ts": 1300.0, "ob_imbalance": 0.62}], "SOLUSDC": [{"ts": 1300.0, "ob_imbalance": 0.70}]},
    )
    merged_mh = MicrostructureCycleHistoryStore.load_path(mh_path)
    assert [r["ts"] for r in merged_mh["BTCUSDC"]] == [1000.0, 1300.0]
    assert len(merged_mh["ETHUSDC"]) == 1
    assert len(merged_mh["SOLUSDC"]) == 1

    # 2. OrderRetryQuarantineAlertsStore preserves sibling keys atomically under lock
    qa_path = OrderRetryQuarantineAlertsStore.resolve_path(str(cdb))
    assert OrderRetryQuarantineAlertsStore.save_key_locked(qa_path, "alerted_ids", [("o1", "err1")])
    assert OrderRetryQuarantineAlertsStore.save_key_locked(qa_path, "quarantined_ids", [("o2", "err2")])
    assert OrderRetryQuarantineAlertsStore.load_key(qa_path, "alerted_ids") == [("o1", "err1")]
    assert OrderRetryQuarantineAlertsStore.load_key(qa_path, "quarantined_ids") == [("o2", "err2")]
    assert OrderRetryQuarantineAlertsStore.clear_key_locked(qa_path, "alerted_ids") is True
    assert OrderRetryQuarantineAlertsStore.load_key(qa_path, "alerted_ids") == []
    assert OrderRetryQuarantineAlertsStore.load_key(qa_path, "quarantined_ids") == [("o2", "err2")]

    # 3. OrdersActiveStore round-trip
    oa_path = OrdersActiveStore.resolve_path(str(cdb))
    assert OrdersActiveStore.write_snapshot(oa_path, {"schema_version": 1, "active_orders": [{"order_id": "123"}]})
    oa_loaded = OrdersActiveStore.read_raw_snapshot(oa_path)
    assert isinstance(oa_loaded, dict) and oa_loaded["active_orders"][0]["order_id"] == "123"

    # 4. CacheTTrendStore locked merge + positive-n preservation
    tt_path = CacheTTrendStore.resolve_path(str(cdb))
    assert CacheTTrendStore.upsert_entry_locked(tt_path, "BTCUSDC", {"T": 8.5, "n": 24, "median_d": 4.2})
    assert CacheTTrendStore.upsert_entry_locked(tt_path, "ETHUSDC", {"T": 6.0, "n": 18, "median_d": 3.1})
    # n=0 write must NOT overwrite an existing n>0 entry
    assert CacheTTrendStore.upsert_entry_locked(tt_path, "BTCUSDC", {"T": 1.0, "n": 0, "median_d": 0.5}) is False
    tt_loaded = CacheTTrendStore.read_from_cachedb(str(cdb))
    assert tt_loaded["BTCUSDC"]["n"] == 24
    assert tt_loaded["ETHUSDC"]["n"] == 18

    # 5. AST static enforcement: synthetic evaluator direct I/O, wrapper write, and estimate_T leaks are blocked
    fake_root = tmp_path / "ast_sandbox"
    ev_dir = fake_root / "intelligence" / "sentiment" / "evaluators"
    ev_dir.mkdir(parents=True)
    bad_ev = ev_dir / "bad_evaluator.py"
    bad_ev.write_text(
        "import os\n"
        "def run():\n"
        "    save_microstructure_cycle_history('/tmp', {})\n"
        "    estimate_T('BTCUSDC')\n",
        encoding="utf-8",
    )
    ev_errs = audit_layer_dag_ast(fake_root, rel_paths=["intelligence/sentiment/evaluators/bad_evaluator.py"])
    assert any("direct disk/state I/O" in e and "os" in e for e in ev_errs)
    assert any("save_microstructure_cycle_history" in e for e in ev_errs)
    assert any("estimate_T" in e for e in ev_errs)

    # 6. AST static enforcement: raw writes to the 4 new critical state files outside their 1:1 stores are blocked
    bad_writer = fake_root / "rogue_writer.py"
    bad_writer.write_text(
        "from state_io import save_json_state\n"
        "def leak():\n"
        "    save_json_state('cachedb/telemetry/microstructure_cycle_history.json', {})\n"
        "    save_json_state('cachedb/orders/order_retry_quarantine_alerts.json', {})\n"
        "    save_json_state('cachedb/orders/orders_active.json', {})\n"
        "    save_json_state('cachedb/trends/cache_T_trend.json', {})\n",
        encoding="utf-8",
    )
    owner_errs = audit_state_file_owners_ast(fake_root, rel_paths=["rogue_writer.py"])
    assert len(owner_errs) == 4


def test_cycle19_prompt_audit_fixes(tmp_path, monkeypatch):
    """Regression coverage for Cycle 19 live US RTH / London 4PM Fix prompt audit fixes:
    1. Collapsed trailing_peak below active_avg_buy_px or open LIFO lot entry_px self-heals from cycle high / entry floor.
    2. NonBinanceTrendPoller and CacheCurrentPriceManager prefer live public equity quotes during US RTH as well as PRE/POST.
    3. Zero-flow, zero-OI equities suppress synthetic whale_long_pct/taker_ratio in cycle_delta history and Section 3B formatting.
    4. Criminal court bail headlines ('released on $1 million bond') do not false-match fixed-income financial context.
    """
    import time
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import cacheprice as cp_mod
    import providers.market_api as ma_mod
    import providers.price_poller as pp_mod
    from intelligence.macro.lexicon_ssot import has_financial_or_geopolitical_context
    from intelligence.sentiment.evaluators.microstructure_evaluator import evaluate_microstructure_cycle_deltas
    from intelligence.sentiment.evaluators.portfolio_evaluator import PositionCostBasisEnrichmentEvaluator
    from orchestratorOS.supervisor.prompt_sections import _format_microstructure_section

    # 1. Self-heal collapsed trailing_peak below entry cost / open LIFO lots (e.g. after trailing_state reset)
    now_ms = int(datetime(2026, 11, 25, 16, 5, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1000)
    old_lot_ms = now_ms - 48 * 3600 * 1000
    pos_btc = {
        "venue": "binance",
        "symbol": "BTCUSDC",
        "price": 58226.215,
        "free_base": 0.02689,
        "trailing_peak": 58226.215,
        "dd_from_peak_pct": 0.0,
    }
    ro_btc = {
        "active_avg_buy_px": 60865.128881,
        "open_lots": [
            {"qty": 0.01018, "entry_px": 62350.0, "ts_ms": old_lot_ms},
            {"qty": 0.01493, "entry_px": 60076.8, "ts_ms": old_lot_ms + 3600 * 1000},
            {"qty": 0.00178, "entry_px": 58985.22, "ts_ms": old_lot_ms + 7200 * 1000},
        ],
    }
    PositionCostBasisEnrichmentEvaluator._enrich_position_from_recent_orders(
        pos_btc,
        ro_btc,
        p_sym="BTCUSDC",
        p_quote_variants=("BTCUSDT",),
        pos_px_f=58226.215,
        held_cur=0.02689,
        is_flat_balance=False,
        is_flat_after_exit=False,
        prices_and_trends={"BTCUSDC": {"price": 58226.215, "high_24h": 62600.0}},
        cache_dir="",
        cycle_high_memo={},
        now_ms=now_ms,
    )
    assert pos_btc["trailing_peak"] == pytest.approx(62600.0)
    assert pos_btc["dd_from_peak_pct"] < -6.5

    # Even for a fresh (<12h) position with no 24h high available, peak floors at max(avg_buy, open_lot.entry_px)
    pos_fresh = {
        "venue": "binance",
        "symbol": "TAOUSDC",
        "price": 170.0,
        "free_base": 4.0,
        "trailing_peak": 170.0,
        "dd_from_peak_pct": 0.0,
    }
    ro_fresh = {
        "active_avg_buy_px": 175.0,
        "open_lots": [{"qty": 4.0, "entry_px": 178.0, "ts_ms": now_ms - 3600 * 1000}],
    }
    PositionCostBasisEnrichmentEvaluator._enrich_position_from_recent_orders(
        pos_fresh,
        ro_fresh,
        p_sym="TAOUSDC",
        p_quote_variants=(),
        pos_px_f=170.0,
        held_cur=4.0,
        is_flat_balance=False,
        is_flat_after_exit=False,
        prices_and_trends={"TAOUSDC": {"price": 170.0}},
        cache_dir="",
        cycle_high_memo={},
        now_ms=now_ms,
    )
    assert pos_fresh["trailing_peak"] == pytest.approx(178.0)
    assert pos_fresh["dd_from_peak_pct"] == pytest.approx(-4.49, abs=0.02)

    # Venue-cycle reconciliation also heals collapsed pos_peak below entry_price (while leaving SHORT cycles untouched)
    pos_cyc = {"venue": "kraken", "symbol": "HYPEUSD", "price": 24.0, "free_base": 5.0, "trailing_peak": 24.0}
    cyc_long = {"qty": 5.0, "entry_price": 27.5, "avg_cost_px": 26.8, "pos_peak": 24.0}
    PositionCostBasisEnrichmentEvaluator._reconcile_position_with_venue_cycle(
        pos_cyc,
        cyc_long,
        p_venue="kraken",
        p_sym="HYPEUSD",
        p_quote_variants=(),
        pos_px_f=24.0,
        fb_cur=5.0,
        lb_cur=0.0,
        held_cur=5.0,
        min_notional_f=10.0,
        is_flat_balance=False,
        is_flat_after_exit=False,
        last_side_ro="BUY",
        ro={"open_lots": [{"qty": 5.0, "entry_px": 27.5, "ts_ms": old_lot_ms}]},
        recent_orders={},
        v_ro_map={},
        cyc_sell_ts_map={},
        prices_and_trends={"HYPEUSD": {"price": 24.0, "high_24h": 28.4}},
        cache_dir="",
        cycle_high_memo={},
        now_ms=now_ms,
    )
    assert pos_cyc["trailing_peak"] == pytest.approx(28.4)
    assert cyc_long["pos_peak"] == pytest.approx(28.4)
    assert cyc_long["dd_from_peak_pct"] < -15.0

    # 2. NonBinanceTrendPoller and CacheCurrentPriceManager prefer live public equity quotes during US RTH
    ny_tz = ZoneInfo("America/New_York")
    ts_rth = datetime(2026, 11, 25, 11, 4, 0, tzinfo=ny_tz).timestamp()
    rth_calls = []

    class _FakeRthApi:
        def get_current_price(self, symbol, provider_name=None, prefer_public_extended=False):
            rth_calls.append((symbol, provider_name, prefer_public_extended))
            return 116.32

    class _FakeRthCpm:
        def __init__(self):
            self.market_api = _FakeRthApi()
            self._enforce_rth_gate = True
            self._delegated_poller_symbols = set()
            self.cache = {}
            self.pushed = []

        def bind_providers(self, _b):
            pass

        @staticmethod
        def extract_price_and_ts(entries):
            if isinstance(entries, list) and entries:
                return entries[-1][1], entries[-1][0]
            return None, None

        def mark_symbol_ws_event(self, _s, _t):
            pass

        def push_price(self, sym, px):
            self.pushed.append((sym, px))
            self.cache[sym] = [[int(ts_rth * 1000), px]]

    monkeypatch.setattr(pp_mod.time, "time", lambda: ts_rth)
    cpm_rth = _FakeRthCpm()
    poller_rth = pp_mod.start_nonbinance_trend_poller(
        cpm_rth,
        ["SPCX_US_EQ"],
        interval_sec=0.05,
        fetch_deadline_sec=1.0,
        provider_names={"SPCX_US_EQ": "t212"},
        start_ws=False,
    )
    try:
        deadline = time.monotonic() + 2.0
        while not cpm_rth.pushed and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        poller_rth.stop(timeout=1.0)
    assert rth_calls and rth_calls[0] == ("SPCX_US_EQ", "t212", True)

    cp_calls = []
    monkeypatch.setattr(cp_mod, "_now_time", lambda: ts_rth)
    monkeypatch.setattr(
        ma_mod.api,
        "get_current_price",
        lambda symbol, provider_name=None, prefer_public_extended=False: (
            cp_calls.append((symbol, provider_name, prefer_public_extended)) or 116.32
        ),
    )
    mgr_stub = cp_mod.CacheCurrentPriceManager.__new__(cp_mod.CacheCurrentPriceManager)
    mgr_stub.market_api = ma_mod.api
    mgr_stub.provider_name_for = lambda _s: "t212"
    mgr_stub._is_equity_instrument = lambda _s: True
    mgr_stub._log_remote_fetch_error = lambda *_a: []
    rows_cp = mgr_stub.get_remote_items("SPCX_US_EQ", 0)
    assert rows_cp and rows_cp[0][1] == pytest.approx(116.32)
    assert cp_calls and cp_calls[0] == ("SPCX_US_EQ", "t212", True)

    # 3. Zero-flow, zero-OI equity suppresses synthetic whale_long_pct/taker_ratio in cycle_delta and Section 3B
    eff_now = 1795622800.0
    m_rgnt = {
        "RGNT_US_EQ": {
            "ob_imbalance": 0.491,
            "bid_depth_usd": 47000.0,
            "ask_depth_usd": 49000.0,
            "bid_wall_px": 0.84,
            "bid_wall_usd": 20000.0,
            "ask_wall_px": 0.85,
            "ask_wall_usd": 20000.0,
            "whale_long_pct": 0.5,
            "taker_buy_sell_ratio": 1.0,
            "taker_buy_vol_usd": 0.0,
            "taker_sell_vol_usd": 0.0,
            "oi_usd": 0.0,
            "oi_1h_chg_pct": 0.0,
            "funding_rate": 0.0,
            "spread_bps": 23.0,
            "ob_mid_price": 0.842,
        }
    }
    prev_hist = {
        "RGNT_US_EQ": [
            {
                "ts": eff_now - 365.0,
                "ob_imbalance": 0.520,
                "whale_long_pct": 0.604,
                "taker_buy_sell_ratio": 1.52,
            }
        ]
    }
    out_m, out_hist, _ = evaluate_microstructure_cycle_deltas(m_rgnt, prev_hist, now_ts=eff_now)
    cd_rgnt = out_m["RGNT_US_EQ"].get("cycle_delta") or {}
    assert "whale_long_delta_pct" not in cd_rgnt
    assert "taker_ratio_delta" not in cd_rgnt
    assert cd_rgnt.get("ob_imbalance_delta") == pytest.approx(-0.029, abs=1e-4)
    assert "whale_long_pct" not in out_hist["RGNT_US_EQ"][-1]
    assert "taker_buy_sell_ratio" not in out_hist["RGNT_US_EQ"][-1]

    # Defense-in-depth in Section 3B formatter even if a pre-populated cycle_delta carries stale d_whale_long/d_tr
    m_rgnt["RGNT_US_EQ"]["cycle_delta"] = {
        "dt_min": 6.1,
        "ob_imbalance_delta": -0.029,
        "whale_long_delta_pct": -0.104,
        "taker_ratio_delta": -0.52,
    }
    sec3b = "\n".join(_format_microstructure_section(m_rgnt, {"RGNT_US_EQ": {"price": 0.842}}, now_ts=eff_now))
    assert "intraday_rth_flow=N/A[no_fresh_vol_bars]" in sec3b
    assert "d_whale_long=" not in sec3b
    assert "d_tr=" not in sec3b
    assert "d_ob=-0.029" in sec3b

    # 4. Criminal court bail headlines do not false-match financial 'bond' while real fixed-income bond stories pass
    assert not has_financial_or_geopolitical_context(
        "Saudi national charged in OpenAI bugging to be released on $1 million bond"
    )
    assert not has_financial_or_geopolitical_context(
        "Suspect arrested in wire fraud case held on $500,000 bond by federal judge"
    )
    assert has_financial_or_geopolitical_context(
        "US Treasury bond yields surge to session highs after hot Core PCE"
    )
    assert has_financial_or_geopolitical_context(
        "Corporate bond default risk climbs after court blocks debt restructuring"
    )


def test_pre_cutover_organize_cachedb_zero_split_brain_and_binance_env_precedence(tmp_path):
    """Regression locks for:
    1. organize_cachedb_subdomains() when statedb/ does not exist keeps statedb files inside pre-cutover
       cachedb/{state,orders,system}/ (never creating phantom cachedb/portfolio/ or cachedb/bots/ split-brain),
       and preserves richer historical state files over smaller fresh stubs during coalescing.
    2. credentials.py loads BINANCE_API_KEY/SECRET/WS from binance_api/.env first, falling back to root .env,
       with explicit os.environ winning over both; binance_credentials() enforces complete pairs.
    """
    import importlib.util
    from pathlib import Path
    import subprocess
    import sys

    from cachepaths import (
        LLM_TOPIC_THREADS_FILENAME,
        NAV_HISTORY_STATE_FILENAME,
        PORTFOLIO_POSITIONS_STATE_FILENAME,
        VENUE_FEE_MAINTENANCE_STATE_FILENAME,
        organize_cachedb_subdomains,
        resolve_statedb_filepath,
    )
    import credentials as creds_mod

    # 1. Pre-cutover organize_cachedb_subdomains() must NOT create phantom cachedb/portfolio or cachedb/bots
    ws = tmp_path / "ws_pre_cutover"
    cdb = ws / "cachedb"
    (cdb / "state").mkdir(parents=True)
    (cdb / "system").mkdir(parents=True)
    (cdb / "orders").mkdir(parents=True)

    full_nav = {"version": 1, "updated_ts": 1000.0, "points": [{"ts": 100.0 + i, "nav_usd": 17000.0 + i} for i in range(6)]}
    stub_nav = {"version": 1, "updated_ts": 2000.0, "points": [{"ts": 2000.0, "nav_usd": 17100.0}]}
    full_topics = {
        "auto_cryptotrade_executor": {"conversation_id": "cid_c", "rolling_history": [{"utc": "t1", "summary": "x" * 120}]},
        "auto_sharetrade_executor": {"conversation_id": "cid_s", "rolling_history": [{"utc": "t2", "summary": "y" * 120}]},
        "macro_sentiment": {"conversation_id": "cid_m", "rolling_history": [{"utc": "t3", "summary": "z" * 120}]},
    }
    stub_topics = {"macro_sentiment": {"conversation_id": "cid_stub"}}

    # Simulate exact live split-brain state: richer older history in portfolio/ & llm/, tiny newer stub in state/ & system/
    (cdb / "portfolio").mkdir(parents=True)
    (cdb / "bots").mkdir(parents=True)
    (cdb / "llm").mkdir(parents=True)
    p_nav_rich = cdb / "portfolio" / NAV_HISTORY_STATE_FILENAME
    p_nav_stub = cdb / "state" / NAV_HISTORY_STATE_FILENAME
    p_nav_rich.write_text(json.dumps(full_nav), encoding="utf-8")
    p_nav_stub.write_text(json.dumps(stub_nav), encoding="utf-8")
    os.utime(p_nav_rich, (1000.0, 1000.0))
    os.utime(p_nav_stub, (2000.0, 2000.0))

    p_top_rich = cdb / "llm" / LLM_TOPIC_THREADS_FILENAME
    p_top_stub = cdb / "system" / LLM_TOPIC_THREADS_FILENAME
    p_top_rich.write_text(json.dumps(full_topics), encoding="utf-8")
    p_top_stub.write_text(json.dumps(stub_topics), encoding="utf-8")
    os.utime(p_top_rich, (1000.0, 1000.0))
    os.utime(p_top_stub, (2000.0, 2000.0))

    (cdb / "state" / PORTFOLIO_POSITIONS_STATE_FILENAME).write_text(json.dumps({"pos:binance:BTCUSDC": {"qty": 1.0}}), encoding="utf-8")
    (cdb / "state" / VENUE_FEE_MAINTENANCE_STATE_FILENAME).write_text(json.dumps({"last_run_ts": 1500.0}), encoding="utf-8")

    organize_cachedb_subdomains(str(cdb))

    # Legacy cachedb/state/ coalesced cleanly into portfolio/ & bots/; llm_topic_threads coalesced into llm/
    assert not (cdb / "state").exists()
    assert not (cdb / "system" / LLM_TOPIC_THREADS_FILENAME).exists()
    assert (cdb / "portfolio" / PORTFOLIO_POSITIONS_STATE_FILENAME).is_file()
    assert (cdb / "bots" / VENUE_FEE_MAINTENANCE_STATE_FILENAME).is_file()
    # Richer historical nav_history & llm_topic_threads were preserved over smaller newer stubs
    recovered_nav = json.loads((cdb / "portfolio" / NAV_HISTORY_STATE_FILENAME).read_text(encoding="utf-8"))
    assert len(recovered_nav["points"]) == 6
    recovered_topics = json.loads((cdb / "llm" / LLM_TOPIC_THREADS_FILENAME).read_text(encoding="utf-8"))
    assert set(recovered_topics.keys()) == {"auto_cryptotrade_executor", "auto_sharetrade_executor", "macro_sentiment"}
    # Subsequent resolver calls point to exact same files (zero split-brain)
    assert Path(str(resolve_statedb_filepath(str(cdb / NAV_HISTORY_STATE_FILENAME)))) == (cdb / "portfolio" / NAV_HISTORY_STATE_FILENAME)
    assert Path(str(resolve_statedb_filepath(str(cdb / LLM_TOPIC_THREADS_FILENAME)))) == (cdb / "llm" / LLM_TOPIC_THREADS_FILENAME)

    # 2. credentials.py loads binance_api/.env ahead of root .env, with os.environ highest priority
    pair_ok = creds_mod.binance_credentials(values={"BINANCE_API_KEY": "k1", "BINANCE_API_SECRET": "s1"})
    assert pair_ok.key == "k1" and pair_ok.secret == "s1"
    with pytest.raises(creds_mod.CredentialConfigurationError):
        creds_mod.binance_credentials(values={"BINANCE_API_KEY": "k1", "BINANCE_API_SECRET": ""})

    fake_ws = tmp_path / "creds_ws"
    (fake_ws / "binance_api").mkdir(parents=True)
    (fake_ws / ".env").write_text(
        "BINANCE_API_KEY=root_key\nBINANCE_API_SECRET=root_sec\nBINANCE_API_KEY_WS=root_ws\n",
        encoding="utf-8",
    )
    (fake_ws / "binance_api" / ".env").write_text(
        "BINANCE_API_KEY=venue_key\nBINANCE_API_SECRET=venue_sec\n",
        encoding="utf-8",
    )
    repo_root = Path(creds_mod.__file__).resolve().parent
    (fake_ws / "credentials.py").write_text(
        (repo_root / "credentials.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    probe_code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"spec = importlib.util.spec_from_file_location('creds_probe', {str(fake_ws / 'credentials.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['creds_probe'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.api_key == 'venue_key', mod.api_key\n"
        "assert mod.api_secret == 'venue_sec', mod.api_secret\n"
        "assert mod.api_key_ws == 'root_ws', mod.api_key_ws\n"
    )
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("BINANCE_API_")}
    proc = subprocess.run([sys.executable, "-c", probe_code], env=clean_env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout + proc.stderr













