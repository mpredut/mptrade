"""Unit and CLI tests for standalone multi-venue AI attribution, model_used audit, and override evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import pytest

from providers.ai_attribution import (
    evaluate_ai_attribution,
    format_ai_attribution_report,
    load_and_evaluate_ai_attribution,
)
from verify_tools import ai_attribution_report as attr_cli


def _make_sample_fixture(now_s: float):
    intents = [
        # 1. Binance BTCUSDC BUY -> SELL winning round-trip on gemini-4-neon-high with plan TARGET_HIT + G_PROF override (AI_RIGHT)
        {
            "intent_id": "ai_1_buy",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-high",
            "clamped_notional_usd": 1000.0,
            "invalidation_px": 68000.0,
            "target_px": 71400.0,
            "advisory_guards_overridden": ["G_PROF:VETO(buying above last sell)"],
            "execution_payload": {"orderId": "bn_ai_1", "status": "FILLED", "executedQty": "0.0142857", "avg_fill_price": 70000.0},
        },
        {
            "intent_id": "ai_1_sell",
            "timestamp": now_s - 3600.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-high",
            "clamped_notional_usd": 1020.0,
            "execution_payload": {"orderId": "bn_ai_2", "status": "FILLED", "executedQty": "0.0142857", "avg_fill_price": 71400.0},
        },
        # 2. Hyperliquid HYPEUSDC BUY -> SELL losing round-trip on gemini-4-neon-medium with plan INVALIDATED + GI_WHALE override on BUY (GUARD_RIGHT)
        #    and G_PROF override on defensive SELL exit (MUST NOT falsely blame G_PROF for the entry loss!)
        {
            "intent_id": "ai_2_buy",
            "timestamp": now_s - 5400.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 49.0,
            "target_px": 53.0,
            "advisory_guards_overridden": ["GI_WHALE:SCALE(heavy taker selling)"],
            "execution_payload": {"orderId": "hl_ai_1", "status": "FILLED", "executedQty": "10.0", "avg_fill_price": 50.0},
        },
        {
            "intent_id": "ai_2_sell",
            "timestamp": now_s - 1800.0,
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 490.0,
            "advisory_guards_overridden": ["G_PROF:VETO(selling below cost basis)"],
            "execution_payload": {"orderId": "hl_ai_2", "status": "FILLED", "executedQty": "10.0", "avg_fill_price": 49.0},
        },
        # 3. Kraken xStock SPCXxUSD open BUY (0 bps fee, unplanned, claude-sonnet-5-5-medium) -> open MTM gain +2.0%
        {
            "intent_id": "ai_3_buy",
            "timestamp": now_s - 4000.0,
            "decision": "BUY",
            "symbol": "SPCXxUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "claude-sonnet-5-5-medium",
            "clamped_notional_usd": 400.0,
            "execution_payload": {"orderId": "kr_ai_1", "status": "FILLED", "executedQty": "40.0", "avg_fill_price": 10.0},
        },
        # 4. Rejected by 4B-1 negative-EV target guard -> tracked in lifecycle + guard trigger stats, never in PnL
        {
            "intent_id": "ai_4_rej",
            "timestamp": now_s - 900.0,
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "status": "REJECTED_AI_CIRCUIT_BREAKER",
            "model_used": "gemini-4-neon-medium",
            "circuit_breaker_reason": "ai_negative_ev_target(TAOUSD@kraken:target=200.5, gross=0.25%<min=0.64%, rt_fee=32bps)",
        },
        # 5. Out-of-scope equity intent on T212 (for scope isolation testing)
        {
            "intent_id": "ai_eq_1",
            "timestamp": now_s - 3000.0,
            "decision": "BUY",
            "symbol": "NVDA_US_EQ",
            "venue": "t212",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-high",
            "clamped_notional_usd": 300.0,
            "execution_payload": {"orderId": "t212_ai_1", "status": "FILLED", "executedQty": "2.0", "avg_fill_price": 150.0},
        },
    ]

    # Multi-venue execution_audit JSONL records (including duplicate order_status polls to verify deduplication)
    audit_records = [
        {"ts": now_s - 7200.0, "event": "submit_accepted", "intent_id": "ex_ai_1", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_ai_1", "side": "buy", "qty": 0.0142857, "price": 70000.0, "client_order_id": "AI_1"},
        {"ts": now_s - 7195.0, "event": "order_status", "intent_id": "ex_ai_1", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_ai_1", "status": "filled", "filled_qty": 0.0142857, "cost": 1000.0, "fee": 1.0},
        {"ts": now_s - 7190.0, "event": "order_status", "intent_id": "ex_ai_1", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_ai_1", "status": "filled", "filled_qty": 0.0142857, "cost": 1000.0, "fee": 1.0},
        {"ts": now_s - 3600.0, "event": "submit_accepted", "intent_id": "ex_ai_2", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_ai_2", "side": "sell", "qty": 0.0142857, "price": 71400.0, "client_order_id": "AI_2"},
        {"ts": now_s - 3595.0, "event": "order_status", "intent_id": "ex_ai_2", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_ai_2", "status": "filled", "filled_qty": 0.0142857, "cost": 1020.0, "fee": 1.0},
        # BOT (tradeall) profitable round-trip on Binance TAOUSDC
        {"ts": now_s - 6000.0, "event": "submit_accepted", "intent_id": "ex_bot_1", "venue": "binance", "symbol": "TAOUSDC", "order_id": "bn_bot_1", "side": "buy", "qty": 5.0, "price": 200.0, "client_order_id": "TA_101"},
        {"ts": now_s - 5995.0, "event": "order_status", "intent_id": "ex_bot_1", "venue": "binance", "symbol": "TAOUSDC", "order_id": "bn_bot_1", "status": "filled", "filled_qty": 5.0, "cost": 1000.0, "fee": 1.0},
        {"ts": now_s - 2000.0, "event": "submit_accepted", "intent_id": "ex_bot_2", "venue": "binance", "symbol": "TAOUSDC", "order_id": "bn_bot_2", "side": "sell", "qty": 5.0, "price": 206.0, "client_order_id": "TA_102"},
        {"ts": now_s - 1995.0, "event": "order_status", "intent_id": "ex_bot_2", "venue": "binance", "symbol": "TAOUSDC", "order_id": "bn_bot_2", "status": "filled", "filled_qty": 5.0, "cost": 1030.0, "fee": 1.0},
        # MANUAL fill on Binance
        {"ts": now_s - 1000.0, "event": "submit_accepted", "intent_id": "ex_mn_1", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_mn_1", "side": "buy", "qty": 0.005, "price": 70000.0, "client_order_id": "and_999"},
        {"ts": now_s - 995.0, "event": "order_status", "intent_id": "ex_mn_1", "venue": "binance", "symbol": "BTCUSDC", "order_id": "bn_mn_1", "status": "filled", "filled_qty": 0.005, "cost": 350.0},
    ]

    ts_lbl = datetime.fromtimestamp(now_s - 1200.0, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [
        f"{ts_lbl} INFO [DELTA GATE] Delta-Gate HOLD carry-forward (quiet_flat); redundant LLM call skipped.",
        f"{ts_lbl} INFO [DELTA GATE] Delta-Gate HOLD carry-forward (quiet_flat); redundant LLM call skipped.",
        f"{ts_lbl} INFO [DELTA GATE] Delta-Gate HOLD carry-forward (quiet_flat); redundant LLM call skipped.",
        f"{ts_lbl} INFO [auto_cryptotrade_executor] Escalating reasoning effort to gemini-4-neon-high for high-signal cycle.",
        f"{ts_lbl} INFO Autonomous AI Supervisor cycle complete: Executed BTCUSDC take profit.",
        f"{ts_lbl} INFO Autonomous AI Supervisor cycle complete: HOLD.",
    ]
    live_prices = {"SPCXxUSD": 10.20, "NVDA_US_EQ": 153.0}
    return intents, audit_records, log_lines, live_prices


def test_ai_attribution_pure_evaluation_and_override_audit():
    """Verify fee-adjusted LIFO PnL, model_used grouping, override accuracy, 4A plan calibration, AI vs BOT, and Delta-Gate stats."""
    now_s = 1_795_300_000.0
    intents, audit_records, log_lines, live_prices = _make_sample_fixture(now_s)

    rep = evaluate_ai_attribution(
        intents,
        now_ts=now_s,
        lookback_days=7.0,
        scope="crypto",
        live_prices=live_prices,
        audit_records=audit_records,
        reconciler_log_lines=log_lines,
        crypto_eval_state={"model_used": "gemini-4-neon-high", "ts": now_s - 300.0},
    )

    # Scope='crypto' excludes NVDA_US_EQ (T212 equity)
    assert rep["ai_intents_total"] == 6
    assert rep["ai_filled_total"] == 5
    assert rep["ai_status_counts"]["EXECUTED_FILLED"] == 5
    assert rep["ai_status_counts"]["REJECTED_AI_CIRCUIT_BREAKER"] == 1
    assert rep["ai_circuit_breaker_triggers"]["ai_negative_ev_target"] == 1

    # Fee-adjusted PnL math:
    # - BTCUSDC Binance: +2.00% gross on $1000 = +$20.00 minus 20bps ($2.00) = +$18.00 net
    # - HYPEUSDC HL: -2.00% gross on $490 = -$9.80 minus 8bps ($0.392) = -$10.192 net
    # - SPCXxUSD Kraken xStock open MTM: +2.00% gross on $400 = +$8.00 minus 0bps ($0.00) = +$8.00 net
    ai_sum = rep["ai_summary"]
    assert ai_sum["closed_trades"] == 2
    assert ai_sum["open_mtm"] == 1
    assert ai_sum["wins"] == 2 and ai_sum["losses"] == 1
    assert ai_sum["realized_gross_usd"] == pytest.approx(10.20, abs=0.05)
    assert ai_sum["realized_net_usd"] == pytest.approx(7.81, abs=0.05)
    assert ai_sum["open_mtm_net_usd"] == pytest.approx(8.00, abs=0.05)
    assert ai_sum["total_net_usd"] == pytest.approx(15.81, abs=0.05)
    assert ai_sum["profit_factor"] == pytest.approx((18.0 + 8.0) / 10.192, abs=0.05)

    # Breakdown by model_used
    by_m = rep["ai_by_model"]
    assert by_m["gemini-4-neon-high"]["realized_net_usd"] == pytest.approx(18.00, abs=0.05)
    assert by_m["gemini-4-neon-medium"]["realized_net_usd"] == pytest.approx(-10.19, abs=0.05)
    assert by_m["claude-sonnet-5-5-medium"]["open_mtm_net_usd"] == pytest.approx(8.00, abs=0.05)

    # Advisory guard override audit: G_PROF scored ONLY on the BTCUSDC BUY entry (AI_RIGHT=1, GUARD_RIGHT=0),
    # NOT falsely blamed for the HYPEUSDC defensive SELL exit!
    ov = rep["advisory_guard_override_audit"]
    assert ov["G_PROF"]["ai_right"] == 1 and ov["G_PROF"]["guard_right"] == 0
    assert ov["G_PROF"]["net_usd"] == pytest.approx(18.00, abs=0.05)
    assert ov["GI_WHALE"]["ai_right"] == 0 and ov["GI_WHALE"]["guard_right"] == 1
    assert ov["GI_WHALE"]["net_usd"] == pytest.approx(-10.19, abs=0.05)

    # 4A Trade Plan Calibration
    plans = rep["trade_plan_calibration"]
    assert plans["planned_filled"] == 2 and plans["unplanned_filled"] == 3
    assert plans["status_counts"]["TARGET_HIT"] == 1
    assert plans["status_counts"]["INVALIDATED"] == 1

    # Multi-venue execution_audit AI vs BOT vs MANUAL (deduplicated polls!)
    actors = rep["execution_audit_ai_vs_bot"]["by_actor"]
    assert actors["AI"]["filled_orders"] == 2 and actors["AI"]["matched_round_trips"] == 1
    assert actors["AI"]["realized_net_usd"] == pytest.approx(18.00, abs=0.05)
    assert actors["BOT"]["filled_orders"] == 2 and actors["BOT"]["matched_round_trips"] == 1
    assert actors["BOT"]["realized_net_usd"] == pytest.approx(28.00, abs=0.05)
    assert actors["MANUAL"]["filled_orders"] == 1
    assert rep["execution_audit_ai_vs_bot"]["by_owner"]["tradeall"]["filled_orders"] == 2

    # Delta-Gate efficiency
    dg = rep["delta_gate_efficiency"]
    assert dg["delta_gate_skips"] == 3
    assert dg["llm_cycles_completed"] == 2
    assert dg["calls_saved_pct"] == pytest.approx(60.0)
    assert dg["high_effort_escalations"] == 1
    assert dg["last_eval_model"] == "gemini-4-neon-high"

    # Equities scope isolates NVDA_US_EQ cleanly
    rep_eq = evaluate_ai_attribution(intents, now_ts=now_s, lookback_days=7.0, scope="equities", live_prices=live_prices)
    assert rep_eq["ai_intents_total"] == 1 and rep_eq["ai_summary"]["open_mtm"] == 1
    assert "t212:NVDA_US_EQ" in rep_eq["ai_by_symbol"]


def test_ai_attribution_workspace_loader_and_cli(tmp_path, capsys):
    """Verify workspace loader and standalone CLI verify_tools/ai_attribution_report.py emit valid JSON and text reports."""
    now_s = datetime.now(timezone.utc).timestamp()
    intents, audit_records, log_lines, live_prices = _make_sample_fixture(now_s)
    day_str = datetime.fromtimestamp(now_s, timezone.utc).strftime("%Y-%m-%d")

    (tmp_path / "cachedb" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cachedb" / "system").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cachedb" / "prices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logger" / "execution_audit").mkdir(parents=True, exist_ok=True)

    (tmp_path / "cachedb" / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps(intents), encoding="utf-8"
    )
    (tmp_path / "cachedb" / "system" / "auto_cryptotrade_executor_eval.json").write_text(
        json.dumps({"ts": now_s - 60.0, "model_used": "gemini-4-neon-high"}), encoding="utf-8"
    )
    (tmp_path / "cachedb" / "prices" / "cache_current_price.json").write_text(
        json.dumps({"items": {"SPCXxUSD": [[int(now_s * 1000), live_prices["SPCXxUSD"]]], "NVDA_US_EQ": live_prices["NVDA_US_EQ"]}}),
        encoding="utf-8",
    )
    (tmp_path / "logger" / "execution_audit" / f"execution_audit_{day_str}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in audit_records) + "\n", encoding="utf-8"
    )
    (tmp_path / "logger" / f"autonomous_ai_reconciler_{day_str}.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    loaded = load_and_evaluate_ai_attribution(str(tmp_path), lookback_days=7.0, scope="all", now_ts=now_s)
    assert loaded["ai_intents_total"] == 7
    txt = format_ai_attribution_report(loaded)
    assert "MPTrade AI Attribution & Audit Report" in txt
    assert "gemini-4-neon-high" in txt
    assert "Advisory Guard Override Audit" in txt
    assert "Multi-Venue Execution Audit (AI vs BOT vs MANUAL)" in txt
    assert "Delta-Gate Efficiency" in txt

    # CLI --json mode emits strict JSON stdout
    rc_json = attr_cli.main(["--workspace", str(tmp_path), "--days", "7", "--scope", "crypto", "--json"])
    assert rc_json == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["scope"] == "crypto"
    assert cli_payload["ai_summary"]["closed_trades"] == 2
    assert cli_payload["delta_gate_efficiency"]["delta_gate_skips"] == 3

    # CLI default text mode emits formatted table
    rc_txt = attr_cli.main(["--workspace", str(tmp_path), "--days", "7"])
    assert rc_txt == 0
    cli_out = capsys.readouterr().out
    assert "=== MPTrade AI Attribution & Audit Report" in cli_out
