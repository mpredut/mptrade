"""Unit and CLI tests for standalone multi-venue AI attribution, model_used audit, and override evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import pytest

from providers.ai_attribution import (
    evaluate_ai_attribution,
    format_ai_attribution_report,
)
from providers.ai_attribution_store import load_and_evaluate_ai_attribution
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
    stale_ts_lbl = datetime.fromtimestamp(now_s - 7.5 * 86400.0, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [
        f"{stale_ts_lbl} INFO Autonomous AI Supervisor cycle complete: Stale 7.5d entry outside 7d window.",
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
    # - HYPEUSDC HL: -2.00% gross on $500 (entry basis) = -$10.00 minus 8bps ($0.40) = -$10.40 net
    # - SPCXxUSD Kraken xStock open MTM: +2.00% gross on $400 = +$8.00 minus 0bps ($0.00) = +$8.00 net
    ai_sum = rep["ai_summary"]
    assert ai_sum["closed_trades"] == 2
    assert ai_sum["open_mtm"] == 1
    assert ai_sum["wins"] == 2 and ai_sum["losses"] == 1
    assert ai_sum["realized_gross_usd"] == pytest.approx(10.00, abs=0.05)
    assert ai_sum["realized_net_usd"] == pytest.approx(7.60, abs=0.05)
    assert ai_sum["open_mtm_net_usd"] == pytest.approx(8.00, abs=0.05)
    assert ai_sum["total_net_usd"] == pytest.approx(15.60, abs=0.05)
    assert ai_sum["profit_factor"] == pytest.approx((18.0 + 8.0) / 10.40, abs=0.05)

    # Breakdown by model_used
    by_m = rep["ai_by_model"]
    assert by_m["gemini-4-neon-high"]["realized_net_usd"] == pytest.approx(18.00, abs=0.05)
    assert by_m["gemini-4-neon-medium"]["realized_net_usd"] == pytest.approx(-10.40, abs=0.05)
    assert by_m["claude-sonnet-5-5-medium"]["open_mtm_net_usd"] == pytest.approx(8.00, abs=0.05)

    # Advisory guard override audit: G_PROF scored ONLY on the BTCUSDC BUY entry (AI_RIGHT=1, GUARD_RIGHT=0),
    # NOT falsely blamed for the HYPEUSDC defensive SELL exit!
    ov = rep["advisory_guard_override_audit"]
    assert ov["G_PROF"]["ai_right"] == 1 and ov["G_PROF"]["guard_right"] == 0
    assert ov["G_PROF"]["net_usd"] == pytest.approx(18.00, abs=0.05)
    assert ov["GI_WHALE"]["ai_right"] == 0 and ov["GI_WHALE"]["guard_right"] == 1
    assert ov["GI_WHALE"]["net_usd"] == pytest.approx(-10.40, abs=0.05)

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

    # CLI default text mode emits formatted table and --out-json atomically persists status JSON
    out_json_path = tmp_path / "cachedb" / "system" / "ai_attribution_status.json"
    rc_txt = attr_cli.main(["--workspace", str(tmp_path), "--days", "7", "--out-json", str(out_json_path)])
    assert rc_txt == 0
    cli_out = capsys.readouterr().out
    assert "=== MPTrade AI Attribution & Audit Report" in cli_out
    assert out_json_path.is_file()
    persisted = json.loads(out_json_path.read_text(encoding="utf-8"))
    assert persisted["ai_intents_total"] == 7

    # Persisted live Binance BNB discount fee (7.5 bps maker = 15 bps RT) and LIMIT execution quality metrics
    from cachepaths import resolve_cache_subdomain

    assert resolve_cache_subdomain("ai_attribution_status.json") == "llm"
    assert resolve_cache_subdomain("ai_attribution_30d.json") == "llm"
    intents_with_lim = [
        {
            **intents[0],
            "limit_price": 70000.0,
            "reference_price": 70350.0,  # filled 50 bps better than arrival spot (-50 bps shortfall)
        },
        *intents[1:],
    ]
    (tmp_path / "cachedb" / "state" / "autonomous_trade_intents_state.json").write_text(
        json.dumps(intents_with_lim), encoding="utf-8"
    )
    (tmp_path / "cachedb" / "state" / "venue_fee_maintenance_state.json").write_text(
        json.dumps({
            "schema_version": 1,
            "ts": now_s - 60.0,
            "state_max_age_sec": 43200.0,
            "venues": {
                "binance": {
                    "ts": now_s - 60.0,
                    "fee_spec": {
                        "fee_tier": "VIP0_BNB25",
                        "maker_fee_bps": 7.5,
                        "taker_fee_bps": 7.5,
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    loaded_bnb = load_and_evaluate_ai_attribution(str(tmp_path), lookback_days=7.0, scope="crypto", now_ts=now_s)
    # BTCUSDC $1000 RT with 15 bps BNB fee = $1.50 fee -> $18.50 net (instead of $18.00 at static 20 bps)
    assert loaded_bnb["ai_by_symbol"]["binance:BTCUSDC"]["realized_net_usd"] == pytest.approx(18.50, abs=0.05)
    assert loaded_bnb["execution_quality"]["limit_filled"] == 5
    assert loaded_bnb["execution_quality"]["avg_limit_slippage_bps"] == pytest.approx(0.0, abs=0.05)
    assert loaded_bnb["execution_quality"]["avg_arrival_shortfall_bps"] == pytest.approx(-49.75, abs=0.5)
    assert "Execution Quality: LIMIT(" in format_ai_attribution_report(loaded_bnb)


def test_execution_audit_monotonic_merge_never_downgrades_fill_on_out_of_order_records():
    """Verify out-of-order submit_accepted record linked by intent_id cannot overwrite an already-merged fill record."""
    now_s = 1_795_000_000.0
    audit_recs = [
        # Pre-oid record created first under intent:i1 (not filled)
        {
            "ts": now_s - 120.0,
            "event": "order_submitted",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "intent_id": "i1",
            "side": "BUY",
            "client_order_id": "AI_i1",
        },
        # Filled record arrives under real order_id without intent_id
        {
            "ts": now_s - 60.0,
            "event": "order_filled",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_id": "ord_999",
            "side": "BUY",
            "filled_qty": 0.01,
            "price": 70000.0,
            "cost": 700.0,
            "client_order_id": "AI_i1",
        },
        # Late non-fill status update linking intent_id + order_id must NOT overwrite is_fill / filled_qty / cost
        {
            "ts": now_s - 30.0,
            "event": "status_checked",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_id": "ord_999",
            "intent_id": "i1",
            "side": "BUY",
            "client_order_id": "AI_i1",
        },
    ]
    rep = evaluate_ai_attribution([], now_ts=now_s, lookback_days=7.0, audit_records=audit_recs)
    ai_actor = rep["execution_audit_ai_vs_bot"]["by_actor"].get("AI")
    assert ai_actor is not None
    assert ai_actor["filled_orders"] == 1
    assert ai_actor["buy_notional_usd"] == pytest.approx(700.0)


def test_ai_attribution_symmetric_sell_trade_plan_calibration():
    """Verify unmatched planned SELL trims are scored in trade_plan_calibration without distorting spot ai_summary."""
    now_s = 1_795_300_000.0
    intents = [
        # Planned SELL hitting lower re-buy target ($184.20) -> TARGET_HIT, positive MTM avoided drawdown
        {
            "intent_id": "ai_sell_hit",
            "timestamp": now_s - 3600.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 600.0,
            "invalidation_px": 191.5,
            "target_px": 184.2,
            "execution_payload": {"orderId": "bn_s1", "status": "FILLED", "executedQty": "3.2", "avg_fill_price": 187.5},
        },
        # Planned SELL invalidated by breakout above invalidation_px ($66.40) -> INVALIDATED, negative MTM edge
        {
            "intent_id": "ai_sell_inv",
            "timestamp": now_s - 2400.0,
            "decision": "SELL",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-high",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 66.4,
            "target_px": 64.2,
            "execution_payload": {"orderId": "kr_s1", "status": "FILLED", "executedQty": "7.6", "avg_fill_price": 65.45},
        },
        # Planned SELL without live price -> counted as ACTIVE
        {
            "intent_id": "ai_sell_act",
            "timestamp": now_s - 1800.0,
            "decision": "SELL",
            "symbol": "ADAUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 300.0,
            "invalidation_px": 0.16,
            "target_px": 0.14,
            "execution_payload": {"orderId": "kr_s2", "status": "FILLED", "executedQty": "2000.0", "avg_fill_price": 0.15},
        },
    ]
    rep = evaluate_ai_attribution(
        intents,
        now_ts=now_s,
        lookback_days=7.0,
        scope="crypto",
        live_prices={"TAOUSDC": 183.0, "HYPEUSD": 67.0},
    )
    # Spot ai_summary remains untouched by unmatched cash SELL exits
    assert rep["ai_summary"]["scored"] == 0
    # trade_plan_calibration scores all 3 planned SELL intents symmetrically
    plans = rep["trade_plan_calibration"]
    assert plans["planned_filled"] == 3
    assert plans["status_counts"] == {"TARGET_HIT": 1, "INVALIDATED": 1, "ACTIVE": 1, "CLOSED_EARLY": 0}
    p_perf = plans["planned_performance"]
    assert p_perf["scored"] == 2
    assert p_perf["closed_trades"] == 0
    assert p_perf["open_mtm"] == 2
    assert p_perf["wins"] == 1 and p_perf["losses"] == 1
    assert p_perf["win_rate_pct"] == pytest.approx(50.0)


def test_delta_gate_log_summary_tracks_wake_reasons():
    """Verify summarize_delta_gate_logs and format_ai_attribution_report surface Delta-Gate wake reasons."""
    from providers.ai_attribution import summarize_delta_gate_logs

    logs = [
        "2026-11-20 12:00:01 [INFO] [DELTA GATE] Delta-Gate HOLD carry-forward (quiet_flat); redundant LLM call skipped.",
        "2026-11-20 12:06:01 [INFO] [DELTA GATE] Wake LLM (price_move(HYPEUSDC:0.82%>=0.60%), high_effort=False).",
        "2026-11-20 12:06:15 [INFO] Autonomous AI Supervisor cycle complete: HOLD",
        "2026-11-20 12:12:01 [INFO] [DELTA GATE] Wake LLM (open_orders_changed, high_effort=False).",
        "2026-11-20 12:12:14 [INFO] Autonomous AI Supervisor cycle complete: HOLD",
        "2026-11-20 12:18:01 [INFO] [DELTA GATE] Wake LLM (price_move(BTCUSDC:0.50%>=0.35%), high_effort=False).",
        "2026-11-20 12:18:16 [INFO] Autonomous AI Supervisor cycle complete: HOLD",
    ]
    dg = summarize_delta_gate_logs(logs)
    assert dg["delta_gate_skips"] == 1
    assert dg["llm_cycles_completed"] == 3
    assert dg["wake_reasons"] == {"price_move": 2, "open_orders_changed": 1}

    rep = evaluate_ai_attribution([], now_ts=1_795_300_000.0, reconciler_log_lines=logs)
    rendered = format_ai_attribution_report(rep)
    assert "Wake reasons: price_move=2, open_orders_changed=1" in rendered


def test_faza4_prep_shared_ssot_primitives(monkeypatch):
    """Verify Faza 4 prep SSOT helpers across cachepaths, survival guard, attribution TCA, and quota router."""
    from cachepaths import (
        AI_ATTRIBUTION_30D_FILENAME,
        AI_ATTRIBUTION_STATUS_FILENAME,
        LLM_QUOTA_STATUS_FILENAME,
        resolve_cache_subdomain,
    )
    from intelligence.sentiment.llm_quota_router import (
        evaluate_model_quota_escalation_allowed,
        resolve_quota_low_warn_fraction,
        summarize_quota_pool_remaining,
    )
    from providers.ai_attribution import (
        extract_fill_execution_slippage,
        format_execution_quality_parts,
        summarize_execution_quality_bucket,
    )
    from providers.ai_survival_guard import (
        build_intent_live_price_map,
        compute_signed_price_edge_pct,
        extract_live_prices_from_cache_payload,
        format_signed_usd,
        match_filled_intent_rows_lifo,
        resolve_matched_notional_usd,
    )

    # 1. LLM cache filename SSOT constants
    for fn in (AI_ATTRIBUTION_STATUS_FILENAME, AI_ATTRIBUTION_30D_FILENAME, LLM_QUOTA_STATUS_FILENAME):
        assert resolve_cache_subdomain(fn) == "llm"

    # 2. Pure LIFO matching, signed edge, USD formatting, and live-price payload helpers
    assert compute_signed_price_edge_pct("BUY", 100.0, 102.5) == pytest.approx(2.5)
    assert compute_signed_price_edge_pct("SELL", 100.0, 97.5) == pytest.approx(2.5)
    assert compute_signed_price_edge_pct("BUY", 0.0, 100.0) == 0.0
    assert resolve_matched_notional_usd(500.0, 300.0) == 300.0
    assert resolve_matched_notional_usd(500.0, 0.0) == 500.0
    assert format_signed_usd(12.345) == "+$12.35"
    assert format_signed_usd(-12.345, decimals=0) == "-$12"

    rows = [
        {"ts": 20.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "SELL", "px": 102.0, "usd": 500.0},
        {"ts": 10.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "BUY", "px": 100.0, "usd": 500.0},
        {"ts": 30.0, "inst_key": ("kraken", "ETH", ("USD",)), "side": "BUY", "px": 2000.0, "usd": 400.0},
        {"ts": 40.0, "inst_key": ("kraken", "TAO", ("USD",)), "side": "SELL", "px": 200.0, "usd": 300.0},
    ]
    closed, open_b, open_s = match_filled_intent_rows_lifo(rows)
    assert len(closed) == 1 and closed[0][0] == "BUY->SELL" and closed[0][1]["ts"] == 10.0 and closed[0][2]["ts"] == 20.0
    assert len(open_b) == 1 and open_b[0]["ts"] == 30.0
    assert len(open_s) == 1 and open_s[0]["ts"] == 40.0

    parsed_px = extract_live_prices_from_cache_payload({
        "items": {
            "BTCUSDC": 70000.0,
            "ETHUSDC": [[1_795_000_000.0, 2500.0]],
            "TAOUSDC": {"price": 190.5},
            "BAD": -5.0,
        }
    })
    assert parsed_px == {"BTCUSDC": 70000.0, "ETHUSDC": 2500.0, "TAOUSDC": 190.5}
    live_map = build_intent_live_price_map(
        {"BTCUSDC": {"price": 70000.0}},
        [{"venue": "kraken", "symbol": "TAOUSD", "price": 190.5}],
    )
    assert live_map["BTCUSDC"] == {"price": 70000.0}
    assert live_map["TAOUSD"] == 190.5
    assert live_map["kraken:TAOUSD"] == 190.5

    # 3. Modular TCA execution quality helpers (global + per-venue slicing)
    lim_s, arr_s, mkt_s = extract_fill_execution_slippage(
        {"limit_price": 100.0, "reference_price": 100.5}, side="BUY", exec_px=99.9, order_type="LIMIT"
    )
    assert lim_s == pytest.approx(-10.0, abs=0.05)
    assert arr_s == pytest.approx(-59.7, abs=0.1)
    assert mkt_s is None
    lim_s2, arr_s2, mkt_s2 = extract_fill_execution_slippage(
        {"reference_price": 100.0}, side="SELL", exec_px=99.8, order_type="MARKET"
    )
    assert lim_s2 is None and arr_s2 is None and mkt_s2 == pytest.approx(20.0, abs=0.05)

    eq_bucket = summarize_execution_quality_bucket([
        {"order_type": "LIMIT", "limit_slip_bps": lim_s, "arrival_sf_bps": arr_s, "market_slip_bps": None},
        {"order_type": "MARKET", "limit_slip_bps": None, "arrival_sf_bps": None, "market_slip_bps": mkt_s2},
    ])
    assert eq_bucket["limit_filled"] == 1 and eq_bucket["market_filled"] == 1
    parts = format_execution_quality_parts(eq_bucket)
    assert len(parts) == 2 and parts[0].startswith("LIMIT(n=1") and parts[1].startswith("MARKET(n=1")
    assert format_execution_quality_parts({"limit_filled": 0, "market_filled": 0}) == []

    # 4. Shared quota pool & escalation policy helpers
    monkeypatch.setenv("LLM_QUOTA_LOW_WARN_FRACTION", "0.15")
    assert resolve_quota_low_warn_fraction() == pytest.approx(0.15)
    monkeypatch.setenv("LLM_QUOTA_LOW_WARN_FRACTION", "0.0")
    assert resolve_quota_low_warn_fraction() == pytest.approx(0.0)
    monkeypatch.setenv("LLM_QUOTA_LOW_WARN_FRACTION", "0.10")

    q_snap = {
        "available_credits": 0.0,
        "buckets": {
            "gemini-4-neon-high": 0.08,
            "claude-sonnet-5-5-medium": 0.99,
            "gemini-3.8-flash-low": 1.0,
        },
    }
    pool = summarize_quota_pool_remaining(
        q_snap,
        paid_gemini_models=("gemini-4-neon-high",),
        flash_gemini_models=("gemini-3.8-flash-low",),
        claude_models=("claude-sonnet-5-5-medium",),
    )
    assert pool == {"available_credits": 0.0, "primary_rem": 0.08, "claude_rem": 0.99, "flash_rem": 1.0}
    assert evaluate_model_quota_escalation_allowed(q_snap, "gemini-4-neon-high") == (False, 0.08, 0.10)
    assert evaluate_model_quota_escalation_allowed(q_snap, " GEMINI-4-NEON-HIGH ") == (False, 0.08, 0.10)
    assert evaluate_model_quota_escalation_allowed(q_snap, "claude-sonnet-5-5-medium", min_remaining_fraction=0.995) == (False, 0.99, 0.995)
    assert evaluate_model_quota_escalation_allowed({}, "gemini-4-neon-high") == (True, None, 0.10)
    assert evaluate_model_quota_escalation_allowed({**q_snap, "available_credits": 50.0}, "gemini-4-neon-high") == (True, 0.08, 0.10)
    assert evaluate_model_quota_escalation_allowed(q_snap, "claude-sonnet-5-5-medium") == (True, 0.99, 0.10)

    # 5. Defensive edge cases: lowercase/invalid sides, NaN/non-positive prices, corrupt slippage fields
    assert compute_signed_price_edge_pct("buy", 100.0, 102.5) == pytest.approx(2.5)
    assert compute_signed_price_edge_pct("HOLD", 100.0, 102.5) == 0.0
    assert compute_signed_price_edge_pct("BUY", float("nan"), 100.0) == 0.0
    assert resolve_matched_notional_usd(float("nan"), 250.0) == 250.0
    closed_d, open_bd, open_sd = match_filled_intent_rows_lifo([
        {"ts": 1.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "HOLD", "px": 100.0, "usd": 100.0},
        {"ts": 2.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "buy", "px": 100.0, "usd": 100.0},
        {"ts": 3.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "sell", "px": 101.0, "usd": 100.0},
    ])
    assert len(closed_d) == 1 and len(open_bd) == 0 and len(open_sd) == 0
    assert extract_fill_execution_slippage({"limit_price": 100.0}, side="buy", exec_px=99.9, order_type="limit")[0] == pytest.approx(-10.0, abs=0.05)
    assert extract_fill_execution_slippage({"limit_price": 100.0}, side="HOLD", exec_px=99.9, order_type="LIMIT") == (None, None, None)
    assert extract_fill_execution_slippage({"limit_price": 100.0}, side="BUY", exec_px=float("nan"), order_type="LIMIT") == (None, None, None)
    eq_dirty = summarize_execution_quality_bucket([
        {"order_type": "limit", "limit_slip_bps": float("nan"), "arrival_sf_bps": -5.0},
    ])
    assert eq_dirty["limit_filled"] == 1 and eq_dirty["avg_limit_slippage_bps"] is None and eq_dirty["avg_arrival_shortfall_bps"] == -5.0
    assert format_execution_quality_parts({"limit_filled": 1, "market_filled": 0, "avg_limit_slippage_bps": "nan"}) == ["LIMIT(n=1)"]


def test_faza4_per_venue_order_type_tca_fill_rate_and_quota_logs():
    """Verify Faza 4B per-(venue, order_type) TCA breakdown, LIMIT fill rate, and quota log telemetry."""
    now_s = 1_795_400_000.0
    intents = [
        # Binance LIMIT BUY filled with -10bps limit improvement and -20bps arrival improvement
        {
            "intent_id": "tca_1",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 70000.0,
            "reference_price": 70070.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "execution_payload": {"orderId": "bn_1", "status": "FILLED", "executedQty": "0.0143", "avg_fill_price": 69930.0},
        },
        # Binance LIMIT BUY unfilled (resting/canceled) -> drives 50.0% fill rate on binance:LIMIT
        {
            "intent_id": "tca_2",
            "timestamp": now_s - 5400.0,
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 180.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "bn_2", "status": "NEW", "executedQty": "0.0"},
        },
        # Hyperliquid MARKET SELL filled with +20bps adverse market slippage
        {
            "intent_id": "tca_3",
            "timestamp": now_s - 3600.0,
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "MARKET",
            "reference_price": 50.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 600.0,
            "execution_payload": {"orderId": "hl_1", "status": "FILLED", "executedQty": "12.0", "avg_fill_price": 49.9},
        },
        # Kraken unfilled-only LIMIT order -> still visible in by_venue / by_venue_order_type with 0.0% fill rate
        {
            "intent_id": "tca_4",
            "timestamp": now_s - 1800.0,
            "decision": "BUY",
            "symbol": "ADAUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 0.14,
            "status": "EXECUTED",
            "clamped_notional_usd": 300.0,
            "execution_payload": {"orderId": "kr_1", "status": "CANCELED", "executedQty": "0.0"},
        },
    ]
    logs = [
        "2026-11-20 12:00:01 [INFO] [auto_cryptotrade_executor] Escalating reasoning effort to gemini-4-neon-high for high-signal cycle.",
        "2026-11-20 12:06:01 [INFO] [auto_cryptotrade_executor] Suppressing high-effort escalation to gemini-4-neon-high (suppress_soft_conserve(gemini-4-neon-high:22.0%<=25.0%)).",
        "2026-11-20 12:12:01 [INFO] [auto_cryptotrade_executor] Quota soft-conserve routing routine cycle to claude-sonnet-5-5-medium (soft_conserve_fallback(gemini-4-neon-medium:21.0%<=25.0%->claude-sonnet-5-5-medium:99.0%)).",
        "2026-11-20 12:18:01 [INFO] [auto_cryptotrade_executor] Quota soft-conserve routing routine cycle to openai_compat:glm-5.4 (soft_conserve_fallback(gemini-4-neon-medium:20.0%<=25.0%->openai_compat:glm-5.4)).",
        "2026-11-20 12:12:15 [INFO] Autonomous AI Supervisor cycle complete: HOLD",
    ]
    rep = evaluate_ai_attribution(intents, now_ts=now_s, lookback_days=7.0, scope="crypto", reconciler_log_lines=logs)
    eq = rep["execution_quality"]
    assert eq["limit_filled"] == 1
    assert eq["limit_unfilled"] == 2
    assert eq["limit_fill_rate_pct"] == pytest.approx(33.3, abs=0.1)
    assert eq["market_filled"] == 1
    assert eq["by_venue"]["binance"]["limit_filled"] == 1
    assert eq["by_venue"]["binance"]["limit_unfilled"] == 1
    assert eq["by_venue"]["binance"]["limit_fill_rate_pct"] == pytest.approx(50.0)
    assert eq["by_venue"]["kraken"]["limit_filled"] == 0
    assert eq["by_venue"]["kraken"]["limit_unfilled"] == 1
    assert eq["by_venue"]["kraken"]["limit_fill_rate_pct"] == pytest.approx(0.0)

    vot = eq["by_venue_order_type"]
    assert vot["binance:LIMIT"]["filled"] == 1 and vot["binance:LIMIT"]["unfilled"] == 1
    assert vot["binance:LIMIT"]["fill_rate_pct"] == pytest.approx(50.0)
    assert vot["binance:LIMIT"]["avg_slippage_bps"] == pytest.approx(-10.0, abs=0.05)
    assert vot["binance:LIMIT"]["avg_arrival_shortfall_bps"] == pytest.approx(-19.98, abs=0.1)
    assert vot["hyperliquid:MARKET"]["filled"] == 1 and vot["hyperliquid:MARKET"]["avg_slippage_bps"] == pytest.approx(20.0, abs=0.05)
    assert vot["kraken:LIMIT"]["filled"] == 0 and vot["kraken:LIMIT"]["unfilled"] == 1

    dg = rep["delta_gate_efficiency"]
    assert dg["high_effort_escalations"] == 1
    assert dg["quota_escalation_suppressions"] == 1
    assert dg["quota_soft_conserve_shifts"] == 2
    assert dg["soft_shift_models"] == {"claude-sonnet-5-5-medium": 1, "openai_compat:glm-5.4": 1}

    rendered = format_ai_attribution_report(rep)
    assert "Execution TCA by Venue:OrderType" in rendered
    assert "binance:LIMIT" in rendered and "hyperliquid:MARKET" in rendered and "kraken:LIMIT" in rendered
    assert "fill_rate=33.3%" in rendered
    assert "quota_suppressed=1, soft_shifts=2" in rendered


def test_faza4_closed_loop_calibration_feedback_and_section_3c_prompt():
    """Verify Faza 4C 0ms BUY/SELL calibration + venue TCA evaluation, scope isolation, and Section 3C rendering."""
    from orchestratorOS.supervisor.prompt_sections import _format_portfolio_section
    from providers.ai_survival_guard import (
        evaluate_ai_calibration_feedback,
        format_ai_calibration_feedback_line,
    )

    now_s = 1_795_400_000.0
    intents = [
        # Closed BUY -> SELL hitting BUY target_px -> BUY TARGET_HIT
        {
            "intent_id": "cal_b1",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 70000.0,
            "reference_price": 70100.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "invalidation_px": 68500.0,
            "target_px": 71400.0,
            "execution_payload": {"orderId": "c1", "status": "FILLED", "executedQty": "0.0143", "avg_fill_price": 70000.0},
        },
        {
            "intent_id": "cal_s1",
            "timestamp": now_s - 5000.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 71500.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 1020.0,
            "execution_payload": {"orderId": "c2", "status": "FILLED", "executedQty": "0.0143", "avg_fill_price": 71500.0},
        },
        # Open planned SELL invalidated by live price breakout -> SELL INVALIDATED
        {
            "intent_id": "cal_s2",
            "timestamp": now_s - 3600.0,
            "decision": "SELL",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "limit_price": 190.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 194.0,
            "target_px": 182.0,
            "execution_payload": {"orderId": "c3", "status": "FILLED", "executedQty": "2.63", "avg_fill_price": 190.0},
        },
        # Canceled Binance LIMIT intent -> 2/3 (67%) fill rate
        {
            "intent_id": "cal_u1",
            "timestamp": now_s - 1800.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 69000.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "c4", "status": "CANCELED", "executedQty": "0.0"},
        },
        # Fresh resting Binance LIMIT intent placed 5m ago -> MUST NOT penalize fill rate while actively working
        {
            "intent_id": "cal_u_fresh",
            "timestamp": now_s - 300.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 69500.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "c4_fresh", "status": "NEW", "executedQty": "0.0"},
        },
        # Out-of-scope equities fill (must stay isolated from scope="crypto")
        {
            "intent_id": "cal_eq1",
            "timestamp": now_s - 1200.0,
            "decision": "BUY",
            "symbol": "NVDA_US_EQ",
            "venue": "t212",
            "order_type": "MARKET",
            "reference_price": 150.0,
            "status": "EXECUTED",
            "clamped_notional_usd": 400.0,
            "invalidation_px": 145.0,
            "target_px": 160.0,
            "execution_payload": {"orderId": "c5", "status": "FILLED", "executedQty": "2.66", "avg_fill_price": 150.15},
        },
    ]

    cal_crypto = evaluate_ai_calibration_feedback(
        intents, now_ts=now_s, live_prices={"TAOUSD": 196.0}, lookback_days=7.0, scope="crypto"
    )
    assert cal_crypto["filled_total"] == 3
    assert cal_crypto["planned_filled"] == 2
    assert cal_crypto["by_side"]["BUY"]["TARGET_HIT"] == 1
    assert cal_crypto["by_side"]["SELL"]["INVALIDATED"] == 1
    assert cal_crypto["resolved_hit_rate_pct"] == pytest.approx(50.0)
    assert "t212" not in cal_crypto["venue_tca"]
    assert cal_crypto["venue_tca"]["binance"]["limit_filled"] == 2
    assert cal_crypto["venue_tca"]["binance"]["limit_unfilled"] == 1

    cal_line = format_ai_calibration_feedback_line(cal_crypto)
    assert cal_line.startswith("AI Execution & Plan Calibration (7d): ")
    assert "plans=2 (BUY:1T/0I/0E/0A, SELL:0T/1I/0E/0A, hit=50%)" in cal_line
    assert "binance: LIM 2/3[67%]" in cal_line
    assert "kraken: LIM 1" in cal_line
    assert "t212" not in cal_line
    assert format_ai_calibration_feedback_line({}) == ""
    assert format_ai_calibration_feedback_line(evaluate_ai_calibration_feedback([], now_ts=now_s)) == ""

    # Section 3C prompt integration renders calibration directly alongside AI Decision Scorecard
    sec_3c = "\n".join(
        _format_portfolio_section(
            {
                "ai_decision_scorecard": {
                    "count": 2,
                    "wins": 1,
                    "scratches": 0,
                    "losses": 1,
                    "avg_edge_pct": 0.35,
                    "net_usd": 12.0,
                    "loss_streak": 0,
                    "status": "HEALTHY",
                },
                "ai_calibration_feedback": cal_crypto,
            },
            asset_scope="crypto",
        )
    )
    assert "AI Decision Scorecard (48h, n=2):" in sec_3c
    assert cal_line in sec_3c


def test_planned_reentry_exit_row_plans_are_evaluated_consistently():
    """Verify SELL->BUY re-entry plans where BOTH legs carry trade plans are evaluated without dropping the BUY leg."""
    from providers.ai_survival_guard import (
        evaluate_ai_calibration_feedback,
        format_ai_calibration_feedback_line,
    )

    now_s = 1_795_200_000.0
    intents = [
        {
            "intent_id": "re_s1",
            "timestamp": now_s - 7200.0,
            "decision": "SELL",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 184.4,
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 187.5,
            "target_px": 182.11,
            "execution_payload": {"orderId": "r1", "status": "FILLED", "executedQty": "2.71", "avg_fill_price": 184.4},
        },
        {
            "intent_id": "re_b1",
            "timestamp": now_s - 3600.0,
            "decision": "BUY",
            "symbol": "TAOUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "limit_price": 178.8,
            "status": "EXECUTED",
            "clamped_notional_usd": 500.0,
            "invalidation_px": 175.8,
            "target_px": 184.4,
            "execution_payload": {"orderId": "r2", "status": "FILLED", "executedQty": "2.79", "avg_fill_price": 178.8},
        },
    ]
    live_px = {"TAOUSDC": 175.1}
    cal = evaluate_ai_calibration_feedback(intents, now_ts=now_s, live_prices=live_px, lookback_days=7.0, scope="crypto")
    assert cal["planned_filled"] == 2
    assert sum(cal["status_counts"].values()) == 2
    assert cal["by_side"]["SELL"]["TARGET_HIT"] == 1
    assert cal["by_side"]["BUY"]["INVALIDATED"] == 1
    assert cal["resolved_hit_rate_pct"] == pytest.approx(50.0)
    line = format_ai_calibration_feedback_line(cal)
    assert "plans=2 (BUY:0T/1I/0E/0A, SELL:1T/0I/0E/0A, hit=50%)" in line

    attr = evaluate_ai_attribution(intents, now_ts=now_s, live_prices=live_px, lookback_days=7.0, scope="crypto")
    plans = attr["trade_plan_calibration"]
    assert plans["planned_filled"] == 2
    assert plans["status_counts"] == {"TARGET_HIT": 1, "INVALIDATED": 1, "ACTIVE": 0, "CLOSED_EARLY": 0}


def test_crypto_executor_prompt_coherence_active_plans_calibration_xstocks_and_deribit_budget(monkeypatch, tmp_path):
    """Verify cross-section prompt coherence fixes for active trade plans, calibration, xStock T212 siblings, and Deribit budget."""
    import intelligence.sentiment.collectors.macro_telemetry_collector as mtc_mod
    from intelligence.sentiment.evaluators.prompt_view_evaluator import PortfolioPromptViewEvaluator
    from providers.ai_survival_guard import evaluate_ai_calibration_feedback
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import summarize_active_ai_trade_plans

    now_s = 1_795_220_000.0

    # 1. Active AI Trade Plans:
    #    - Resting SELL LIMIT on BTCUSDC must NOT suppress the held FILLED BUY position's plan
    #    - Resting plan distances anchor to entry_px (limit price), while FILLED plans anchor to spot_px
    #    - Resolved flat FILLED SELL plans expire after 24h without resurrecting closed prior BUY plans
    plan_intents = [
        {
            "intent_id": "old_closed_buy",
            "timestamp": now_s - 40 * 3600.0,
            "decision": "BUY",
            "venue": "kraken",
            "symbol": "TAOUSD",
            "order_type": "LIMIT",
            "limit_price": 180.0,
            "target_px": 192.0,
            "invalidation_px": 174.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "K-OLD-BUY", "status": "FILLED", "price": 180.0, "filled_qty": 1.0},
        },
        {
            "intent_id": "stale_resolved_sell",
            "timestamp": now_s - 30 * 3600.0,
            "decision": "SELL",
            "venue": "kraken",
            "symbol": "TAOUSD",
            "order_type": "LIMIT",
            "limit_price": 187.4,
            "target_px": 184.2,
            "invalidation_px": 192.5,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "K-STALE-SELL", "status": "FILLED", "price": 187.4, "filled_qty": 1.0},
        },
        {
            "intent_id": "btc_filled_buy",
            "timestamp": now_s - 7200.0,
            "decision": "BUY",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_type": "LIMIT",
            "limit_price": 70000.0,
            "target_px": 72100.0,
            "invalidation_px": 68600.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "B-BTC-BUY", "status": "FILLED", "price": 70000.0, "filled_qty": 0.04},
        },
        {
            "intent_id": "btc_resting_tp_sell",
            "timestamp": now_s - 1800.0,
            "decision": "SELL",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_type": "LIMIT",
            "limit_price": 71500.0,
            "target_px": 69355.0,
            "invalidation_px": 72930.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "B-BTC-REST", "status": "NEW", "price": 71500.0, "executedQty": "0.0"},
        },
    ]
    positions = [
        {"venue": "binance", "symbol": "BTCUSDC", "price": 69500.0, "value_usd": 2780.0, "effective_max_buy_usd": 2000.0, "effective_max_sell_usd": 2000.0},
        {"venue": "kraken", "symbol": "TAOUSD", "price": 175.0, "value_usd": 0.0, "effective_max_buy_usd": 2000.0, "effective_max_sell_usd": 0.0},
    ]
    open_orders = [
        {"venue": "binance", "symbol": "BTCUSDC", "side": "SELL", "order_id": "B-BTC-REST", "price": 71500.0, "status": "NEW"}
    ]
    active_plans = summarize_active_ai_trade_plans(
        plan_intents,
        "crypto",
        now_s,
        extract_intent_notional,
        tradable_positions=positions,
        open_orders=open_orders,
        prices_and_trends={"BTCUSDC": {"price": 69500.0}, "TAOUSD": {"price": 175.0}},
    )
    assert all(p["symbol"] != "TAOUSD" for p in active_plans)
    btc_by_state = {p["order_state"]: p for p in active_plans if p["symbol"] == "BTCUSDC"}
    assert set(btc_by_state) == {"FILLED", "RESTING"}
    assert btc_by_state["FILLED"]["side"] == "BUY"
    assert btc_by_state["FILLED"]["target_dist_pct"] == pytest.approx(((72100.0 - 69500.0) / 69500.0) * 100.0, abs=0.02)
    assert btc_by_state["RESTING"]["side"] == "SELL"
    assert btc_by_state["RESTING"]["target_dist_pct"] == pytest.approx(((69355.0 - 71500.0) / 71500.0) * 100.0, abs=0.02)
    assert btc_by_state["RESTING"]["invalidation_dist_pct"] == pytest.approx(((72930.0 - 71500.0) / 71500.0) * 100.0, abs=0.02)

    # 2. Calibration & Attribution:
    #    - Sell-only Kraken xStock (SPCXxUSD) FILLED SELL exits do not inflate directional SELL plan hit-rate
    #    - Chronological intermediate same-side fill breaching invalidation_px resolves INVALIDATED even if spot later reverses
    cal_intents = [
        {
            "intent_id": "xs_sell_1",
            "timestamp": now_s - 10800.0,
            "decision": "SELL",
            "venue": "kraken",
            "symbol": "SPCXxUSD",
            "order_type": "LIMIT",
            "limit_price": 130.0,
            "target_px": 128.0,
            "invalidation_px": 132.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "XS-1", "status": "FILLED", "price": 130.0, "filled_qty": 2.0},
        },
        {
            "intent_id": "tao_s1",
            "timestamp": now_s - 7200.0,
            "decision": "SELL",
            "venue": "binance",
            "symbol": "TAOUSDC",
            "order_type": "LIMIT",
            "limit_price": 180.0,
            "target_px": 174.0,
            "invalidation_px": 184.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "T-S1", "status": "FILLED", "avg_fill_price": 180.0, "executedQty": "2.0"},
        },
        {
            "intent_id": "tao_s2_breach",
            "timestamp": now_s - 3600.0,
            "decision": "SELL",
            "venue": "binance",
            "symbol": "TAOUSDC",
            "order_type": "LIMIT",
            "limit_price": 186.0,
            "status": "EXECUTED",
            "execution_payload": {"order_id": "T-S2", "status": "FILLED", "avg_fill_price": 186.0, "executedQty": "1.0"},
        },
    ]
    cal_res = evaluate_ai_calibration_feedback(
        cal_intents, now_ts=now_s, live_prices={"SPCXxUSD": 126.0, "TAOUSDC": 172.0}, lookback_days=7.0, scope="crypto"
    )
    assert cal_res["filled_total"] == 3
    assert cal_res["planned_filled"] == 1
    assert cal_res["by_side"]["SELL"]["INVALIDATED"] == 1
    assert cal_res["by_side"]["SELL"]["TARGET_HIT"] == 0
    assert cal_res["venue_tca"]["kraken"]["limit_filled"] == 1
    attr_res = evaluate_ai_attribution(
        cal_intents, now_ts=now_s, live_prices={"SPCXxUSD": 126.0, "TAOUSDC": 172.0}, lookback_days=7.0, scope="crypto"
    )
    assert attr_res["trade_plan_calibration"]["planned_filled"] == 1
    assert attr_res["trade_plan_calibration"]["status_counts"]["INVALIDATED"] == 1

    # 3. PortfolioPromptViewEvaluator surfaces funded non-primary T212 sibling cycles (SPCX_US_EQ) on scope="crypto"
    view = PortfolioPromptViewEvaluator().evaluate(
        {
            "tradable_positions": [
                {"venue": "kraken", "symbol": "SPCXxUSD", "price": 127.8, "free_base": 0.0, "value_usd": 0.0},
                {"venue": "kraken", "symbol": "ANTxUSD", "price": 87.5, "free_base": 0.0, "value_usd": 0.0},
            ],
            "kraken_xstocks": {
                "underlying": "ANT",
                "t212_sibling": "ANT_US_EQ",
                "t212_sibling_cycle": {"qty": 0.0, "cost_usd": 0.0},
            },
            "cross_market_context": {
                "equity_bellwethers": {
                    "SPCX_US_EQ": {
                        "price": 127.8,
                        "chg_24h_pct": -1.5,
                        "t212_cycle": {"qty": 4.442965, "cost_usd": 635.94, "avg_cost_px": 143.1341, "dd_pct": -10.8},
                    }
                }
            },
        },
        asset_scope="crypto",
    )
    xs_payload = view["kraken_xstocks"]["payload"]
    assert "other_t212_sibling_cycles" in xs_payload
    assert xs_payload["other_t212_sibling_cycles"]["SPCX_US_EQ"]["cost_usd"] == pytest.approx(635.94)

    # 4. MacroBenchmarksSnapshotCollector reserves post-Yahoo wall-clock budget so slow Yahoo responses cannot starve Deribit
    import time

    fake_clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_clock[0])

    def _slow_yahoo_fast_deribit(url, timeout=3.5, headers=None):
        if "yahoo" in url:
            fake_clock[0] += 3.0
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {"regularMarketPrice": 580.0, "previousClose": 575.0, "marketState": "REGULAR"},
                            "timestamp": [int(now_s - 3600), int(now_s - 60)],
                            "indicators": {"quote": [{"open": [575.0, 579.0], "high": [576.0, 581.0], "low": [574.0, 578.0], "close": [575.5, 580.0], "volume": [1000, 1200]}]},
                        }
                    ]
                }
            }
        if "coinbase" in url:
            return {"price": "70050.0"}
        if "get_volatility_index_data" in url:
            return {"result": {"data": [[int(now_s * 1000), 51.0, 52.0, 50.5, 51.4]]}}
        if "get_historical_volatility" in url:
            return {"result": [[int(now_s * 1000), 46.2]]}
        if "get_book_summary_by_currency" in url:
            return {"result": []}
        return {}

    monkeypatch.setattr(mtc_mod, "fetch_public_json", _slow_yahoo_fast_deribit)
    snap = mtc_mod.MacroBenchmarksSnapshotCollector().collect(
        cachedb_dir=str(tmp_path), now_ts=now_s, allow_network=True, persist=False
    )
    assert snap is not None
    assert snap.get("btc_dvol") == pytest.approx(51.4)


def test_crypto_executor_prompt_data_accuracy_guards(tmp_path):
    """Verify prompt data accuracy fixes: plan anchor labels, flow-scaled wall TCA, 5m VP impulse tolerance, one-sided hit tag, L1/proxy OI tag, and crypto venue scope."""
    import json
    from types import SimpleNamespace
    from botcore import topic_spec
    from intelligence.sentiment.collectors.microstructure_collector import upgrade_tpo_vp_from_ohlc
    from intelligence.sentiment.evaluators.tca_risk_evaluator import enrich_open_limit_orders_tca
    from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
    from orchestratorOS.supervisor.prompt_sections import (
        _format_microstructure_section,
        _format_portfolio_section,
    )
    from providers.ai_survival_guard import format_ai_calibration_feedback_line

    # 1. Active AI Trade Plans explicitly label vs_spot (+ spot=...) on FILLED vs vs_limit on RESTING
    sec3c = "\n".join(
        _format_portfolio_section(
            {
                "active_ai_trade_plans": [
                    {
                        "venue": "kraken",
                        "symbol": "TAOUSD",
                        "side": "SELL",
                        "order_state": "FILLED",
                        "dist_anchor": "spot",
                        "entry_px": 187.42,
                        "spot_px": 178.38,
                        "target_px": 184.2,
                        "target_dist_pct": 3.26,
                        "invalidation_px": 192.5,
                        "invalidation_dist_pct": 7.92,
                        "age_h": 11.2,
                        "status": "TARGET_HIT",
                    },
                    {
                        "venue": "binance",
                        "symbol": "BTCUSDC",
                        "side": "SELL",
                        "order_state": "RESTING",
                        "dist_anchor": "limit",
                        "entry_px": 60050.0,
                        "spot_px": 59247.69,
                        "target_px": 58300.0,
                        "target_dist_pct": -2.91,
                        "invalidation_px": 61650.0,
                        "invalidation_dist_pct": 2.66,
                        "age_h": 0.1,
                        "status": "ACTIVE",
                    },
                ],
                "kraken_xstocks": {"underlying": "ANT", "t212_sibling": "ANT_US_EQ", "listed_pair": "ANTxUSD"},
            },
            asset_scope="crypto",
        )
    )
    assert "SELL@187.42 (FILLED, spot=178.38, age=11.2h): target=184.2 (+3.26% vs_spot)" in sec3c
    assert "SELL@60050.0 (RESTING, age=0.1h): target=58300.0 (-2.91% vs_limit)" in sec3c
    assert "API-TRADABLE 24/7 on 'kraken' at 0bps maker/taker fee [SELL-ONLY EU API]" in sec3c

    # 2. OpenLimitOrdersTcaEvaluator ignores sub-8% BTC noise quote ($230K on $4.41M depth / $7.95M 15m vol) while keeping real walls
    btc_oo = [{"venue": "binance", "symbol": "BTCUSDC", "side": "SELL", "price": 60050.0}]
    enrich_open_limit_orders_tca(
        btc_oo,
        {"BTCUSDC": {"price": 59247.69, "vol_1h_pct": 1.0}},
        {
            "BTCUSDC": {
                "ob_mid_price": 59233.685,
                "best_bid": 59233.68,
                "best_ask": 59233.69,
                "ask_depth_usd": 4_405_564.0,
                "spot_vol_15m_usd": 7_950_000.0,
                "ask_wall_px": 59414.34,
                "ask_wall_usd": 230_840.0,
            }
        },
    )
    assert "wall_queue_pos" not in btc_oo[0]
    assert "recommended_front_run_px" not in btc_oo[0]

    # 3. upgrade_tpo_vp_from_ohlc keeps REAL_VOL across +2.2% 5m impulse when OHLC bar tail is fresh (<=30m)
    now_vp = 1_795_211_000.0
    ohlc_pts = [
        [int((now_vp - 20 * 3600 + i * 900) * 1000), 60.0 + (i % 3) * 0.5, 1000.0, 61.5, 59.8, 60.2]
        for i in range(80)
    ]
    (tmp_path / "cache_ohlc_24h_HYPEUSDC.json").write_text(
        json.dumps({"ts": now_vp - 180.0, "symbol": "HYPEUSDC", "points": ohlc_pts}),
        encoding="utf-8",
    )
    m24_hype = {"vp_source": "TPO", "low_24h": 59.7, "high_24h": 62.5}
    upgrade_tpo_vp_from_ohlc(
        "HYPEUSDC",
        m24_hype,
        cur_px=61.85,  # +2.23% above last bar close 60.5 -> previously dropped to TPO (>1.5%)
        now_ts=now_vp,
        s24_is_live_clock=True,
        cache_dir=str(tmp_path),
        allow_network=False,
    )
    assert m24_hype["vp_source"] == "REAL_VOL"

    # 4. Calibration line tags one-sided resolved hit rate (hit=95%[SELL_only])
    cal_line = format_ai_calibration_feedback_line(
        {
            "lookback_days": 7.0,
            "planned_filled": 32,
            "by_side": {
                "BUY": {"TARGET_HIT": 0, "INVALIDATED": 0, "CLOSED_EARLY": 3, "ACTIVE": 5},
                "SELL": {"TARGET_HIT": 19, "INVALIDATED": 1, "CLOSED_EARLY": 0, "ACTIVE": 4},
            },
            "resolved_hit_rate_pct": 95.0,
            "venue_tca": {},
        }
    )
    assert "hit=95%[SELL_only]" in cal_line

    # 5. Section 3B marks perp_flow_src=bn_perp when native L1 OI differs from flow-venue OI
    sec3b = "\n".join(
        _format_microstructure_section(
            {
                "HYPEUSDC": {
                    "ob_imbalance": 0.54,
                    "bid_depth_usd": 8_000_000.0,
                    "ask_depth_usd": 7_000_000.0,
                    "whale_long_pct": 0.69,
                    "taker_buy_sell_ratio": 1.27,
                    "taker_buy_vol_usd": 2_800_000.0,
                    "taker_sell_vol_usd": 2_200_000.0,
                    "oi_usd": 308_802_875.55,
                    "native_l1_oi_usd": 1_165_366_228.40,
                    "oi_1h_chg_pct": 0.89,
                    "funding_annualized_pct": 10.9,
                }
            },
            {"HYPEUSDC": {"price": 61.63}},
        )
    )
    assert "oi_usd=308802875.55 (l1_oi=$1.17B)" in sec3b
    assert "perp_flow_src=bn_perp" in sec3b

    # 6. Crypto prompt instructions omit 't212' from allowed venues and state SELL-only xStocks
    crypto_prompt = build_supervisor_prompt(
        SimpleNamespace(),
        LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0),
        topic_spec("auto_cryptotrade_executor"),
    )
    assert "WHICH venue (`binance`, `kraken`, `hyperliquid`)." in crypto_prompt
    assert "0bps SELL-only xStock SPCXxUSD" in crypto_prompt


def test_critical_path_aware_trade_plans_and_fee_adjusted_reload_scorecard():
    """Verify path-aware trade plan evaluation/expiry, Delta-Gate status preservation, reload->exit LIFO pairing, net-of-fee scorecard, and last N/M label."""
    from orchestratorOS.supervisor.crypto_executor_delta_gate import _evaluate_delta_gate_trade_plans
    from providers.ai_survival_guard import (
        evaluate_ai_calibration_feedback,
        evaluate_ai_decision_scorecard,
        format_ai_decision_scorecard_line,
        match_filled_intent_rows_lifo,
    )
    from providers.intent_state import extract_intent_notional
    from providers.intent_telemetry import (
        evaluate_trade_plan_path_status,
        summarize_active_ai_trade_plans,
    )

    now_s = 1_795_300_000.0

    # 1. Path-aware status resolves rebound-after-hit plans and avoids pre-fill wick contamination on straddling bars
    fill_ts = now_s - 29 * 3600.0
    tao_bars = [
        [int((fill_ts - 600.0) * 1000), 188.0, 100.0, 195.0, 180.0, 187.5],
        # Straddling candle covering [fill_ts - 180s, fill_ts + 120s]: pre-fill high 194 > inv 192.5, close 187.0 inside band
        [int((fill_ts + 120.0) * 1000), 187.0, 120.0, 194.0, 186.0, 188.0],
        # Subsequent fully post-fill candle drops to 165.42 (<= target 184.2), then rebounds to 186.96
        [int((fill_ts + 420.0) * 1000), 167.5, 250.0, 187.0, 165.42, 187.0],
        [int((now_s - 300.0) * 1000), 186.96, 110.0, 187.2, 186.1, 186.5],
    ]
    # Straddling bar alone must NOT falsely invalidate on pre-fill wick
    assert (
        evaluate_trade_plan_path_status(
            "SELL",
            186.96,
            192.5,
            184.2,
            start_ts=fill_ts,
            price_bars=tao_bars[:2],
        )
        == "ACTIVE"
    )
    # Full path hits target 184.2 even though live spot rebounded to 186.96
    assert (
        evaluate_trade_plan_path_status(
            "SELL",
            186.96,
            192.5,
            184.2,
            start_ts=fill_ts,
            price_bars=tao_bars,
        )
        == "TARGET_HIT"
    )

    # 29h-old flat kraken:TAOUSD SELL plan is pruned once path-resolved to TARGET_HIT (>24h expiry)
    stale_tao_intents = [
        {
            "intent_id": "tao_old_sell",
            "timestamp": fill_ts,
            "decision": "SELL",
            "venue": "kraken",
            "symbol": "TAOUSD",
            "order_type": "LIMIT",
            "limit_price": 187.42,
            "target_px": 184.2,
            "invalidation_px": 192.5,
            "status": "EXECUTED",
            "clamped_notional_usd": 650.0,
            "execution_payload": {"order_id": "KR-TAO-1", "status": "FILLED", "avg_fill_price": 187.42, "executedQty": "3.47"},
        }
    ]
    tao_positions = [{"venue": "kraken", "symbol": "TAOUSD", "price": 186.96, "value_usd": 0.0}]
    plans_without_bars = summarize_active_ai_trade_plans(
        stale_tao_intents,
        "crypto",
        now_s,
        extract_intent_notional,
        tradable_positions=tao_positions,
        prices_and_trends={"TAOUSD": {"price": 186.96}},
    )
    assert len(plans_without_bars) == 1 and plans_without_bars[0]["status"] == "ACTIVE"
    plans_with_bars = summarize_active_ai_trade_plans(
        stale_tao_intents,
        "crypto",
        now_s,
        extract_intent_notional,
        tradable_positions=tao_positions,
        prices_and_trends={"TAOUSD": {"price": 186.96}},
        price_bars_by_symbol={"TAOUSD": tao_bars},
    )
    assert plans_with_bars == []

    # 2. Delta-Gate preserves already path-resolved non-resting status even if spot rebounded into the band
    dg_states, dg_reason = _evaluate_delta_gate_trade_plans(
        {
            "active_ai_trade_plans": [
                {
                    "venue": "kraken",
                    "symbol": "TAOUSD",
                    "side": "SELL",
                    "order_state": "FILLED",
                    "status": "TARGET_HIT",
                    "target_px": 184.2,
                    "invalidation_px": 192.5,
                }
            ]
        },
        {"TAOUSD": 186.96},
        {"kraken:TAOUSD:SELL": "TARGET_HIT"},
    )
    assert dg_states == {"kraken:TAOUSD:SELL": "TARGET_HIT"}
    assert dg_reason == ""

    # 3. Path-aware calibration & attribution catch intermediate stop-breach (INVALIDATED vs CLOSED_EARLY) and target-hit rebound
    btc_buy_ts = now_s - 6 * 3600.0
    btc_sell_ts = now_s - 2 * 3600.0
    btc_bars = [
        [int((btc_buy_ts + 300.0) * 1000), 56800.0, 50.0, 57000.0, 56500.0, 56942.0],
        [int((btc_buy_ts + 1800.0) * 1000), 55400.0, 90.0, 56200.0, 55056.0, 56100.0],
        [int((btc_sell_ts - 300.0) * 1000), 58200.0, 70.0, 58300.0, 57900.0, 58000.0],
    ]
    path_intents = [
        {
            "intent_id": "btc_b_path",
            "timestamp": btc_buy_ts,
            "decision": "BUY",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "invalidation_px": 55600.0,
            "target_px": 59500.0,
            "execution_payload": {"orderId": "bn_p1", "status": "FILLED", "avg_fill_price": 56942.0, "executedQty": "0.0175"},
        },
        {
            "intent_id": "btc_s_path",
            "timestamp": btc_sell_ts,
            "decision": "SELL",
            "venue": "binance",
            "symbol": "BTCUSDC",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "clamped_notional_usd": 1022.0,
            "execution_payload": {"orderId": "bn_p2", "status": "FILLED", "avg_fill_price": 58210.0, "executedQty": "0.0175"},
        },
        stale_tao_intents[0],
    ]
    bars_map = {"BTCUSDC": btc_bars, "TAOUSD": tao_bars}
    cal_path = evaluate_ai_calibration_feedback(
        path_intents,
        now_ts=now_s,
        live_prices={"BTCUSDC": 58210.0, "TAOUSD": 186.96},
        price_bars_by_symbol=bars_map,
        lookback_days=7.0,
        scope="crypto",
    )
    assert cal_path["by_side"]["BUY"]["INVALIDATED"] == 1
    assert cal_path["by_side"]["BUY"]["CLOSED_EARLY"] == 0
    assert cal_path["by_side"]["SELL"]["TARGET_HIT"] == 1
    assert cal_path["by_side"]["SELL"]["ACTIVE"] == 0
    attr_path = evaluate_ai_attribution(
        path_intents,
        now_ts=now_s,
        live_prices={"BTCUSDC": 58210.0, "TAOUSD": 186.96},
        price_bars_by_symbol=bars_map,
        lookback_days=7.0,
        scope="crypto",
    )
    assert attr_path["trade_plan_calibration"]["status_counts"] == {
        "TARGET_HIT": 1,
        "INVALIDATED": 1,
        "ACTIVE": 0,
        "CLOSED_EARLY": 0,
    }

    # 4. Reload->exit LIFO pairing, fee-adjusted scorecard PnL, and last N/M window label
    #    - SELL -> BUY (still held) produces only SELL->BUY without duplicate open BUY_MTM
    held_reload = [
        {"ts": 10.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "SELL", "px": 187.42, "usd": 600.0},
        {"ts": 20.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "BUY", "px": 169.50, "usd": 600.0},
    ]
    c_held, ob_held, os_held = match_filled_intent_rows_lifo(held_reload)
    assert len(c_held) == 1 and c_held[0][0] == "SELL->BUY"
    assert ob_held == [] and os_held == []

    #    - SELL -> BUY -> SELL scores BOTH reload (SELL->BUY) and subsequent stop-loss exit (BUY->SELL)
    reload_then_stop = [
        *held_reload,
        {"ts": 30.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "SELL", "px": 166.00, "usd": 588.0},
    ]
    c_stop, ob_stop, os_stop = match_filled_intent_rows_lifo(reload_then_stop)
    assert [p[0] for p in c_stop] == ["SELL->BUY", "BUY->SELL"]
    assert c_stop[1][1]["px"] == 169.50 and c_stop[1][2]["px"] == 166.00
    assert ob_stop == [] and os_stop == []

    #    - Scorecard deducts venue RT fees (marginal +0.15% gross on Binance 20bps RT -> -0.05% net, negative USD)
    #      and renders `last 5/6` when total_outcomes exceeds window_trades=5
    sc_intents = [
        # Outcome 1 (older, dropped from last-5 window): BUY 100 -> SELL 102 (+2.00% gross, +1.80% net)
        {"intent_id": "s1b", "timestamp": now_s - 20000.0, "decision": "BUY", "venue": "binance", "symbol": "BTCUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1000.0, "execution_payload": {"orderId": "o1b", "status": "FILLED", "avg_fill_price": 100.0, "executedQty": "10"}},
        {"intent_id": "s1s", "timestamp": now_s - 19000.0, "decision": "SELL", "venue": "binance", "symbol": "BTCUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1020.0, "execution_payload": {"orderId": "o1s", "status": "FILLED", "avg_fill_price": 102.0, "executedQty": "10"}},
        # Outcome 2: BUY 100 -> SELL 100.15 (+0.15% gross on Binance 20bps RT -> -0.05% net, -$0.50 USD!)
        {"intent_id": "s2b", "timestamp": now_s - 16000.0, "decision": "BUY", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1000.0, "execution_payload": {"orderId": "o2b", "status": "FILLED", "avg_fill_price": 100.0, "executedQty": "10"}},
        {"intent_id": "s2s", "timestamp": now_s - 15000.0, "decision": "SELL", "venue": "binance", "symbol": "ETHUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 1001.5, "execution_payload": {"orderId": "o2s", "status": "FILLED", "avg_fill_price": 100.15, "executedQty": "10"}},
        # Outcomes 3 & 4: SELL@187.42 -> BUY@169.50 (WIN reload) -> SELL@166.00 (LOSS stop-out!)
        {"intent_id": "s3s1", "timestamp": now_s - 12000.0, "decision": "SELL", "venue": "binance", "symbol": "TAOUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 600.0, "execution_payload": {"orderId": "o3s1", "status": "FILLED", "avg_fill_price": 187.42, "executedQty": "3.2"}},
        {"intent_id": "s3b1", "timestamp": now_s - 10000.0, "decision": "BUY", "venue": "binance", "symbol": "TAOUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 600.0, "execution_payload": {"orderId": "o3b1", "status": "FILLED", "avg_fill_price": 169.50, "executedQty": "3.54"}},
        {"intent_id": "s3s2", "timestamp": now_s - 8000.0, "decision": "SELL", "venue": "binance", "symbol": "TAOUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 588.0, "execution_payload": {"orderId": "o3s2", "status": "FILLED", "avg_fill_price": 166.00, "executedQty": "3.54"}},
        # Outcome 5: BUY 100 -> SELL 99.0 (-1.00% gross, -1.20% net)
        {"intent_id": "s4b", "timestamp": now_s - 6000.0, "decision": "BUY", "venue": "binance", "symbol": "ARBUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 500.0, "execution_payload": {"orderId": "o4b", "status": "FILLED", "avg_fill_price": 100.0, "executedQty": "5"}},
        {"intent_id": "s4s", "timestamp": now_s - 4000.0, "decision": "SELL", "venue": "binance", "symbol": "ARBUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 495.0, "execution_payload": {"orderId": "o4s", "status": "FILLED", "avg_fill_price": 99.0, "executedQty": "5"}},
        # Outcome 6: BUY 50 -> SELL 51 (+2.00% gross, +1.92% net on Hyperliquid 8bps RT)
        {"intent_id": "s5b", "timestamp": now_s - 3000.0, "decision": "BUY", "venue": "hyperliquid", "symbol": "HYPEUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 500.0, "execution_payload": {"orderId": "o5b", "status": "FILLED", "avg_fill_price": 50.0, "executedQty": "10"}},
        {"intent_id": "s5s", "timestamp": now_s - 1000.0, "decision": "SELL", "venue": "hyperliquid", "symbol": "HYPEUSDC", "order_type": "LIMIT", "status": "EXECUTED", "clamped_notional_usd": 510.0, "execution_payload": {"orderId": "o5s", "status": "FILLED", "avg_fill_price": 51.0, "executedQty": "10"}},
    ]
    sc_fee = evaluate_ai_decision_scorecard(sc_intents[2:4], now_ts=now_s, window_trades=5, scope="crypto")
    assert sc_fee["count"] == 1 and sc_fee["total_outcomes"] == 1
    assert sc_fee["avg_edge_pct"] == pytest.approx(-0.05, abs=0.01)
    assert sc_fee["net_usd"] == pytest.approx(-0.5, abs=0.05)

    sc_reload_stop = evaluate_ai_decision_scorecard(sc_intents[4:7], now_ts=now_s, window_trades=5, scope="crypto")
    assert sc_reload_stop["count"] == 2 and sc_reload_stop["total_outcomes"] == 2
    assert sc_reload_stop["wins"] == 1 and sc_reload_stop["losses"] == 1 and sc_reload_stop["loss_streak"] == 1

    sc_full = evaluate_ai_decision_scorecard(sc_intents, now_ts=now_s, window_trades=5, scope="crypto")
    assert sc_full["count"] == 5
    assert sc_full["total_outcomes"] == 6
    assert sc_full["wins"] == 2 and sc_full["scratches"] == 1 and sc_full["losses"] == 2
    sc_line = format_ai_decision_scorecard_line(sc_full)
    assert sc_line.startswith("AI Decision Scorecard (48h, last 5/6): 2W/1S/2L,")
