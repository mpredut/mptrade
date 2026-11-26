"""Unit and integration tests for Warm-Turn Multi-Group Delta & Keyframe prompt elision."""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
from intelligence.sentiment.evaluators.prompt_delta_evaluator import (
    PROMPT_DELTA_GROUP_CADENCES,
    evaluate_prompt_delta_plan,
)
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.agy.topics import get_topic_thread_state
from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
from orchestratorOS.supervisor.autonomous_ai_reconciler import AutonomousAIReconciler
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
from verify_tools.audit_prompt_invariants import audit_rendered_prompt_text


def _make_rich_sensors(ts: float, *, anchor_ts: float | None = None) -> TelemetrySensorSnapshot:
    """Build a realistic crypto trading telemetry snapshot covering G0, G1, G2, and G3 blocks."""
    base_ts = anchor_ts if anchor_ts is not None else ts
    snap = TelemetrySensorSnapshot()
    snap.evaluated_at_ts = ts
    snap.source_ages_sec = {"prices": 3.0, "orderbook": 8.0, "macro": 45.0}
    snap.fear_greed = {
        "value": 24,
        "classification": "Extreme Fear",
        "trend_7d_change": -6,
        "historical_values": [24, 26, 28, 30, 31, 29, 30],
        "ts": base_ts - 300.0,
    }
    snap.market_breadth = {
        "advance_ratio": 0.42,
        "median_change_pct": -0.8,
        "mean_change_pct": -0.6,
        "dispersion_std": 2.1,
        "advancing": 12,
        "total_symbols": 29,
        "ts": ts - 60.0,
    }
    snap.sentiment_advisor = {
        "confidence": 0.72,
        "summary": "BTC consolidating near $60.8k POC after ETF outflow stabilization.",
        "live_flow_divergence": "(taker_ratio=0.98, cvd_15m=-$1.2M vs 20.0m-old summary)",
        "ts": base_ts - 1200.0,
    }
    snap.geopolitical_threat = {
        "risk_score": 0.25,
        "summary": "Calm macro backdrop with no acute escalation events.",
        "ts": base_ts - 1800.0,
    }
    snap.macro_calendar = {
        "session_label": "EU/US_OVERLAP",
        "next_high_impact_event": "US Core PCE (2026-11-27 13:30 UTC)",
        "next_event_is_high_impact": True,
        "pre_event_risk_window": False,
        "fomc_today": False,
        "next_fomc_date": "2026-12-16",
    }
    snap.macro_and_sentiment = {
        "etf_flows": {
            "summary_line": "BTC +$125M, ETH +$32M (5d net +$410M)",
            "ts": base_ts - 3600.0,
        },
        "macro_benchmarks": {
            "summary_line": "SPY -0.15%, QQQ -0.22%, VIX 17.4, DXY 104.2",
            "ts": ts - 120.0,
        },
    }
    snap.breaking_news = [
        {"title": "Bitcoin ETF inflows steady ahead of US holiday session", "ts": base_ts - 600.0},
        {"title": "Fed officials reiterate data-dependent path on rates", "ts": base_ts - 1200.0},
        {"title": "Ethereum staking queue remains balanced across validators", "ts": base_ts - 1800.0},
    ]
    snap.prices_and_trends = {
        "BTCUSDC": {
            "price": 60500.0,
            "chg_24h_pct": -0.5,
            "low_24h": 59500.0,
            "high_24h": 61500.0,
            "poc_px": 60200.0,
            "val_px": 59800.0,
            "vah_px": 61100.0,
        },
        "HYPEUSDC": {
            "price": 41.2,
            "chg_24h_pct": 1.8,
            "low_24h": 39.8,
            "high_24h": 42.1,
        },
    }
    snap.microstructure = {
        "BTCUSDC": {
            "venue": "binance",
            "ob_imbalance": 0.542,
            "bid_depth_usd": 450000.0,
            "ask_depth_usd": 380000.0,
            "bid_wall_px": 60350.0,
            "bid_wall_usd": 220000.0,
            "ask_wall_px": 60700.0,
            "ask_wall_usd": 210000.0,
            "spread_bps": 0.4,
            "best_bid": 60499.0,
            "best_ask": 60501.0,
            "ob_mid_price": 60500.0,
            "taker_buy_sell_ratio": 1.04,
            "taker_buy_vol_usd": 1200000.0,
            "taker_sell_vol_usd": 1150000.0,
            "cvd_15m_usd": 50000.0,
            "oi_usd": 1500000000.0,
            "oi_1h_chg_pct": 0.4,
            "_ob_raw_ts": ts - 4.0,
            "_deriv_ts": ts - 20.0,
            "_whale_ts": ts - 25.0,
        },
    }
    snap.portfolio_summary = {
        "portfolio_risk_engine": {
            "unified_nav_usd": 10000.0,
            "invested_exposure_usd": 6000.0,
            "global_free_cash_usd": 4000.0,
            "global_free_cash_pct": 40.0,
            "venue_nav_breakdown_usd": {"binance": 7000.0, "hyperliquid": 3000.0},
            "portfolio_btc_beta": 0.92,
            "avg_corr_vs_btc_24h": 0.85,
            "intraday_var_95_usd": 180.0,
            "intraday_var_95_pct": 1.8,
        },
        "venue_quote_cash": {
            "binance": {"venue": "binance", "asset": "USDC", "free": 2500.0},
            "hyperliquid": {"venue": "hyperliquid", "asset": "USDC", "free": 1500.0},
        },
        "tradable_positions": [
            {
                "venue": "binance",
                "symbol": "BTCUSDC",
                "base": "BTC",
                "quote": "USDC",
                "free_base": 0.08,
                "value_usd": 4840.0,
                "free_quote": 2500.0,
                "price": 60500.0,
                "maker_fee_bps": 0.0,
                "taker_fee_bps": 7.5,
                "fee_tier": "VIP0",
                "min_notional_usd": 10.0,
            },
            {
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "base": "HYPE",
                "quote": "USDC",
                "free_base": 28.0,
                "value_usd": 1153.6,
                "free_quote": 1500.0,
                "price": 41.2,
                "maker_fee_bps": 1.5,
                "taker_fee_bps": 4.5,
                "fee_tier": "T0",
                "min_notional_usd": 10.0,
            },
        ],
        "ai_decision_scorecard": {
            "window_trades": 5,
            "lookback_hours": 48.0,
            "count": 3,
            "total_outcomes": 3,
            "wins": 2,
            "losses": 1,
            "win_rate_pct": 66.7,
            "net_usd": 14.5,
            "avg_net_pct": 0.65,
            "by_side": {"BUY->SELL": {"wins": 2, "losses": 1}},
        },
        "ai_calibration_feedback": {
            "lookback_days": 7.0,
            "planned_filled": 4,
            "resolved_hit_rate_pct": 66.7,
            "by_side": {
                "BUY": {"TARGET_HIT": 2, "INVALIDATED": 1, "CLOSED_EARLY": 0, "ACTIVE": 1},
            },
        },
        "kraken_xstocks": {
            "pair_lbl": "SPCXxUSD",
            "t212_sibling": "SPCX_US_EQ",
            "ipo_lbl": "",
            "payload": {
                "SPCXxUSD": {"qty": 10.0, "avg_cost_px": 15.0, "price": 16.5, "pnl_pct": 10.0},
            },
        },
        "cross_market_context": {
            "equity_bellwethers": {
                "NVDA_US_EQ": {"price": 142.5, "quote_source": "yahoo", "market_state": "REGULAR"},
                "SPCX_US_EQ": {"price": 16.5, "quote_source": "t212", "market_state": "REGULAR"},
            },
        },
        "hyperliquid_readonly": {
            "engine": "hyperliquid_spot",
            "paper": False,
            "qty": 28.0,
            "cash_usd": 1500.0,
        },
        "recent_orders": {
            "BTCUSDC": {
                "last_action_side": "BUY",
                "last_action_actor": "ai",
                "last_buy": {"price": 59900.0, "qty": 0.02, "ts_ms": round((base_ts - 7200.0) * 1000.0)},
                "last_sell": {"price": 61200.0, "qty": 0.01, "ts_ms": round((base_ts - 18000.0) * 1000.0)},
                "open_lots": [{"qty": 0.02, "price": 59900.0}],
                "lifo_realized_pnl_usd": 12.5,
            },
            "HYPEUSDC": {
                "last_action_side": "SELL",
                "last_action_actor": "bot",
                "last_buy": {"price": 39.5, "qty": 10.0, "ts_ms": round((base_ts - 14400.0) * 1000.0)},
                "last_sell": {"price": 41.0, "qty": 5.0, "ts_ms": round((base_ts - 3600.0) * 1000.0)},
                "open_lots": [{"qty": 5.0, "price": 39.5}],
                "lifo_realized_pnl_usd": 7.5,
            },
        },
    }
    return snap


def test_multigroup_cadence_and_transparent_rendering() -> None:
    """Verify G1(3t), G2(5t), and G3(event_only) cadence, transparency markers, and prompt invariants."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    sensors = _make_rich_sensors(t0)
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)

    # Turn 1: Cold keyframe -> all blocks shown, all counters initialized to 0
    plan1 = evaluate_prompt_delta_plan(
        sensors,
        asset_scope="crypto",
        now_ts=t0,
        is_warm_resume=False,
        conversation_id="cid-alpha",
        compacted_at_step=0,
    )
    assert plan1.mode == "FULL_KEYFRAME"
    assert plan1.reason == "cold_turn"
    assert plan1.next_state["blocks"]["macro_calendar"]["turns_omitted"] == 0
    assert plan1.next_state["blocks"]["fear_greed"]["turns_omitted"] == 0

    full_prompt = build_supervisor_prompt(sensors, logs, compact_rules=False, delta_plan=plan1)
    assert "### 0. CLOCK & SNAPSHOT: 2026-11-26 12:00:00 EET (Thu)" in full_prompt
    assert "  * Live: " in full_prompt and "  * Macro: " in full_prompt
    s0_i = full_prompt.index("### 0. CLOCK & SNAPSHOT:")
    s1_i = full_prompt.index("### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:")
    s2_i = full_prompt.index("### 2. MACRO, REGIME & CROSS-MARKET:")
    s3_i = full_prompt.index("### 3. TRADABLE MARKETS (UNIFIED PRICE, BOOK & FLOW \u2014 ALL COINS):")
    assert s0_i < s1_i < s2_i < s3_i
    # Cross-market US equities table lives in Section 2 (Macro & Cross-Market), not Section 1 (Portfolio)
    assert s2_i < full_prompt.index("Cross-Market US Equities Telemetry") < s3_i
    assert "PILLAR 2" not in full_prompt
    # Unified 3-line card under BTCUSDC with Trend / Book / Flow
    assert "    - Trend: " in full_prompt and "    - Book: " in full_prompt and "    - Flow: " in full_prompt
    assert "Raport Execu\u021bie [YYYY-MM-DD HH:MM:SS EET]" in full_prompt
    assert "- Snapshot Mode: FULL_KEYFRAME (reason=cold_turn)" in full_prompt
    assert "Fear & Greed:" in full_prompt
    assert "Live Breaking News Feed" in full_prompt
    assert "BTC consolidating near $60.8k POC" in full_prompt
    assert audit_rendered_prompt_text(full_prompt) == []

    # Turn 2: Unchanged warm turn -> G1 (1/3), G2 (1/5), G3 omitted; G0 + live deltas preserved
    t2 = t0 + 360.0
    sensors_t2 = _make_rich_sensors(t2, anchor_ts=t0)
    plan2 = evaluate_prompt_delta_plan(
        sensors_t2,
        asset_scope="crypto",
        now_ts=t2,
        is_warm_resume=True,
        prev_state=plan1.next_state,
        conversation_id="cid-alpha",
        compacted_at_step=0,
    )
    assert plan2.mode == "WARM_DELTA"
    assert plan2.action_for("macro_calendar") == "OMIT"
    assert plan2.action_for("scorecard") == "OMIT"
    assert plan2.action_for("fear_greed") == "OMIT"
    assert plan2.action_for("etf_flows") == "OMIT"
    assert plan2.action_for("kraken_xstocks") == "OMIT"
    assert plan2.action_for("breaking_news") == "OMIT"
    assert plan2.action_for("sentiment_advisor_summary") == "OMIT"
    assert plan2.action_for("geopolitical_summary") == "OMIT"
    assert plan2.recent_order_actions["BTCUSDC"] == "OMIT"
    assert plan2.recent_order_actions["HYPEUSDC"] == "OMIT"

    warm_prompt_t2 = build_supervisor_prompt(sensors_t2, logs, compact_rules=True, delta_plan=plan2)
    assert "- Snapshot Mode: WARM_DELTA (" in warm_prompt_t2
    assert "- Unchanged in thread (active from prior turn):" in warm_prompt_t2
    assert "Calendar(1/3)" in warm_prompt_t2
    assert "ETF_Flows(1/5)" in warm_prompt_t2
    assert "News(3 prior)" in warm_prompt_t2
    assert "Recent_Fills[2](1/3)" in warm_prompt_t2
    # Elided blocks absent, while live G0 + advisor/geo live deltas remain present
    assert "- Fear & Greed" not in warm_prompt_t2
    assert "Live Breaking News Feed" not in warm_prompt_t2
    assert "BTC consolidating near $60.8k POC" not in warm_prompt_t2
    assert "summary=[unchanged in thread]" in warm_prompt_t2
    assert "live_btc_flow_delta=" in warm_prompt_t2
    assert "Market Breadth:" in warm_prompt_t2
    assert "US Equities, Futures & Macro Rates Benchmarks" in warm_prompt_t2
    assert "Raport Execu\u021bie [YYYY-MM-DD HH:MM:SS EET]" in warm_prompt_t2
    assert len(warm_prompt_t2) < len(full_prompt)
    assert audit_rendered_prompt_text(warm_prompt_t2) == []

    # Compact prefix stays well below the 35% ratio invariant against full prefix
    full_prefix = full_prompt.split("LIVE TELEMETRY SNAPSHOT:", 1)[0]
    compact_prefix = warm_prompt_t2.split("LIVE TELEMETRY SNAPSHOT:", 1)[0]
    assert len(compact_prefix) < len(full_prefix) * 0.35

    # Turn 3: G1 (2/3), G2 (2/5) still OMIT
    t3 = t2 + 360.0
    plan3 = evaluate_prompt_delta_plan(
        _make_rich_sensors(t3, anchor_ts=t0),
        asset_scope="crypto",
        now_ts=t3,
        is_warm_resume=True,
        prev_state=plan2.next_state,
        conversation_id="cid-alpha",
        compacted_at_step=0,
    )
    assert plan3.action_for("macro_calendar") == "OMIT"
    assert plan3.action_for("etf_flows") == "OMIT"
    assert "Calendar(2/3)" in " ".join(plan3.format_section0_lines())

    # Turn 4: G1 hits cadence K=3 -> SHOW_REFRESH & counter resets to 0; G2 is (3/5) OMIT; G3 stays OMIT
    assert PROMPT_DELTA_GROUP_CADENCES["G1"] == 3
    t4 = t3 + 360.0
    sensors_t4 = _make_rich_sensors(t4, anchor_ts=t0)
    plan4 = evaluate_prompt_delta_plan(
        sensors_t4,
        asset_scope="crypto",
        now_ts=t4,
        is_warm_resume=True,
        prev_state=plan3.next_state,
        conversation_id="cid-alpha",
        compacted_at_step=0,
    )
    assert plan4.action_for("macro_calendar") == "SHOW_REFRESH"
    assert plan4.action_for("scorecard") == "SHOW_REFRESH"
    assert plan4.recent_order_actions["BTCUSDC"] == "SHOW_REFRESH"
    assert plan4.action_for("fear_greed") == "OMIT"
    assert plan4.action_for("etf_flows") == "OMIT"
    assert plan4.action_for("breaking_news") == "OMIT"
    assert plan4.action_for("sentiment_advisor_summary") == "OMIT"
    assert plan4.next_state["blocks"]["macro_calendar"]["turns_omitted"] == 0

    warm_prompt_t4 = build_supervisor_prompt(sensors_t4, logs, compact_rules=True, delta_plan=plan4)
    assert "- Macro Economic Calendar & Session Window [REFRESH]:" in warm_prompt_t4
    assert "BTCUSDC Last Action=BUY (by=AI) [REFRESH]:" in warm_prompt_t4
    assert "ETF_Flows(3/5)" in warm_prompt_t4

    # Turn 5: G1 (1/3) OMIT, G2 (4/5) OMIT -> Turn 6: G2 hits K=5 -> SHOW_REFRESH
    t5 = t4 + 360.0
    plan5 = evaluate_prompt_delta_plan(
        _make_rich_sensors(t5, anchor_ts=t0),
        asset_scope="crypto",
        now_ts=t5,
        is_warm_resume=True,
        prev_state=plan4.next_state,
        conversation_id="cid-alpha",
        compacted_at_step=0,
    )
    assert plan5.action_for("fear_greed") == "OMIT"
    assert "Fear&Greed(4/5)" in " ".join(plan5.format_section0_lines())

    t6 = t5 + 360.0
    sensors_t6 = _make_rich_sensors(t6, anchor_ts=t0)
    plan6 = evaluate_prompt_delta_plan(
        sensors_t6,
        asset_scope="crypto",
        now_ts=t6,
        is_warm_resume=True,
        prev_state=plan5.next_state,
        conversation_id="cid-alpha",
        compacted_at_step=0,
    )
    assert plan6.action_for("fear_greed") == "SHOW_REFRESH"
    assert plan6.action_for("etf_flows") == "SHOW_REFRESH"
    assert plan6.action_for("kraken_xstocks") == "SHOW_REFRESH"
    assert plan6.action_for("macro_calendar") == "OMIT"
    warm_prompt_t6 = build_supervisor_prompt(sensors_t6, logs, compact_rules=True, delta_plan=plan6)
    assert "- Fear & Greed [REFRESH]:" in warm_prompt_t6
    assert "- US Spot Crypto ETF Net Flows (Farside Daily) [REFRESH]:" in warm_prompt_t6


def test_local_changes_and_category_coupling() -> None:
    """Verify per-symbol fill delta + execution coupling, macro pre-event coupling, and SpaceX news xstocks coupling."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    base = _make_rich_sensors(t0)
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)
    p1 = evaluate_prompt_delta_plan(base, asset_scope="crypto", now_ts=t0, is_warm_resume=False, conversation_id="cid-1")

    # 1. Single-symbol fill change -> emits ONLY changed symbol + forces execution category (scorecard/calibration/HL)
    t2 = t0 + 300.0
    s_fill = copy.deepcopy(base)
    s_fill.evaluated_at_ts = t2
    s_fill.portfolio_summary["recent_orders"]["BTCUSDC"]["last_buy"]["price"] = 60120.0
    s_fill.portfolio_summary["recent_orders"]["BTCUSDC"]["last_buy"]["ts_ms"] = round((t2 - 60.0) * 1000.0)
    p_fill = evaluate_prompt_delta_plan(
        s_fill, asset_scope="crypto", now_ts=t2, is_warm_resume=True, prev_state=p1.next_state, conversation_id="cid-1"
    )
    assert p_fill.recent_order_actions["BTCUSDC"] == "SHOW_NEW"
    assert p_fill.recent_order_actions["HYPEUSDC"] == "OMIT"
    assert p_fill.action_for("scorecard") == "SHOW_NEW"
    assert p_fill.action_for("calibration") == "SHOW_NEW"
    assert p_fill.action_for("hyperliquid_readonly") == "SHOW_NEW"
    assert p_fill.action_for("fear_greed") == "OMIT"
    txt_fill = build_supervisor_prompt(s_fill, logs, compact_rules=True, delta_plan=p_fill)
    assert "BTCUSDC Last Action=BUY (by=AI) [NEW]:" in txt_fill
    assert "HYPEUSDC Last Action=" not in txt_fill
    assert "Recent_Fills[1](1/3)" in txt_fill

    # 2. Breaking news delta -> emits ONLY new headlines; SpaceX headline couples kraken_xstocks; high geo risk couples geo summary
    s_news = copy.deepcopy(base)
    s_news.evaluated_at_ts = t2
    s_news.geopolitical_threat["risk_score"] = 0.65
    s_news.breaking_news.insert(0, {"title": "SpaceX valuation update triggers sharp xStocks move", "ts": t2 - 30.0})
    p_news = evaluate_prompt_delta_plan(
        s_news, asset_scope="crypto", now_ts=t2, is_warm_resume=True, prev_state=p1.next_state, conversation_id="cid-1"
    )
    assert p_news.action_for("breaking_news") == "SHOW_NEW"
    assert p_news.action_for("kraken_xstocks") == "SHOW_NEW"
    assert p_news.action_for("geopolitical_summary") == "SHOW_NEW"
    txt_news = build_supervisor_prompt(s_news, logs, compact_rules=True, delta_plan=p_news)
    assert "- Live Breaking News Feed [1 NEW; 3 prior unchanged in thread]:" in txt_news
    assert "SpaceX valuation update triggers sharp xStocks move" in txt_news
    assert "Bitcoin ETF inflows steady ahead of US holiday session" not in txt_news
    assert "Kraken Tokenized Equities / xStocks" in txt_news
    assert "context) [NEW]:" in txt_news

    # 3. Pre-event macro risk window forces macro category refresh
    s_macro = copy.deepcopy(base)
    s_macro.evaluated_at_ts = t2
    s_macro.macro_calendar["pre_event_risk_window"] = True
    s_macro.macro_calendar["pre_event_risk_reason"] = "US Core PCE in 15m"
    p_macro = evaluate_prompt_delta_plan(
        s_macro, asset_scope="crypto", now_ts=t2, is_warm_resume=True, prev_state=p1.next_state, conversation_id="cid-1"
    )
    assert p_macro.action_for("macro_calendar") == "SHOW_NEW"
    assert p_macro.action_for("fear_greed") == "SHOW_NEW"
    assert p_macro.action_for("etf_flows") == "SHOW_NEW"
    assert p_macro.action_for("sentiment_advisor_summary") == "SHOW_NEW"
    assert p_macro.action_for("geopolitical_summary") == "SHOW_NEW"


@pytest.mark.parametrize(
    ("trigger_kwargs", "expected_reason"),
    [
        ({"is_warm_resume": False}, "cold_turn"),
        ({"prev_state": {}}, "no_prior_delta_state"),
        ({"conversation_id": "cid-rotated"}, "thread_rotated"),
        ({"compacted_at_step": 12}, "thread_compacted"),
        ({"now_ts": datetime(2026, 11, 27, 0, 5, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()}, "day_rollover"),
        ({"extreme_escalation": True}, "extreme_escalation"),
        ({"urgent_escalation": True}, "urgent_escalation"),
        ({"escalate_high_effort": True}, "high_effort_escalation"),
        ({"ui_turn_pending": True}, "ui_turn"),
        ({"focus_symbols": ["BTCUSDC"]}, "focus_symbols"),
        ({"prev_eval_was_actionable": True}, "post_action"),
    ],
)
def test_global_full_keyframe_triggers_reset_all_counters(
    trigger_kwargs: Dict[str, Any], expected_reason: str
) -> None:
    """Every global invalidation trigger must force FULL_KEYFRAME and zero all cadence counters."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    s0 = _make_rich_sensors(t0)
    p1 = evaluate_prompt_delta_plan(
        s0, asset_scope="crypto", now_ts=t0, is_warm_resume=False, conversation_id="cid-base", compacted_at_step=4
    )
    # Advance one warm delta turn so counters are > 0 in prior state
    t1 = t0 + 360.0
    p2 = evaluate_prompt_delta_plan(
        _make_rich_sensors(t1),
        asset_scope="crypto",
        now_ts=t1,
        is_warm_resume=True,
        prev_state=p1.next_state,
        conversation_id="cid-base",
        compacted_at_step=4,
    )
    assert p2.next_state["blocks"]["fear_greed"]["turns_omitted"] == 1

    call_kwargs: Dict[str, Any] = {
        "asset_scope": "crypto",
        "now_ts": t1 + 360.0,
        "is_warm_resume": True,
        "prev_state": p2.next_state,
        "conversation_id": "cid-base",
        "compacted_at_step": 4,
    }
    call_kwargs.update(trigger_kwargs)
    plan = evaluate_prompt_delta_plan(_make_rich_sensors(call_kwargs["now_ts"]), **call_kwargs)
    assert plan.mode == "FULL_KEYFRAME"
    assert plan.reason == expected_reason
    assert all(b["turns_omitted"] == 0 for b in plan.next_state["blocks"].values())
    assert all(r["turns_omitted"] == 0 for r in plan.next_state["recent_orders"].values())


def test_reconciler_persists_warm_vs_fallback_and_preserves_state_on_failure(tmp_path) -> None:
    """Reconciler persists warm state only when warm resume ran, resets on fallback, and preserves prior state on failure."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    sensors = _make_rich_sensors(t0)
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)

    p1 = evaluate_prompt_delta_plan(
        sensors, asset_scope="crypto", now_ts=t0, is_warm_resume=False, conversation_id="cid-live", compacted_at_step=2
    )

    class _StubClient:
        model = "gemini-test"
        fallback_models = ()
        _explicit_model = True
        last_model_used = "gemini-test"
        last_used_warm_resume = True

        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []
            self.response: Any = {"summary": "HOLD.", "market_trading_decisions": [{"decision": "HOLD", "symbol": "BTCUSDC", "venue": "binance", "confidence": 0.9}]}

        def query_json(self, prompt: str, **kwargs: Any) -> Any:
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            return self.response

    stub = _StubClient()
    reconciler = AutonomousAIReconciler(
        mode="shadow",
        workspace_dir=str(tmp_path),
        gemini_auditor=stub,
        notify_enabled=False,
    )

    # 1. Warm resume succeeded -> persists warm_delta_plan (turns_omitted=1) stamped with post-turn thread state
    with patch(
        "orchestratorOS.supervisor.autonomous_ai_reconciler.get_topic_thread_state",
        side_effect=[("cid-live", 2, True), ("cid-live", 2, True)],
    ):
        reconciler.synthesize_and_diagnose(
            sensors,
            logs,
            topic_key="auto_cryptotrade_executor",
            prev_prompt_delta_state=p1.next_state,
            urgent_escalation=False,
            extreme_escalation=False,
        )
    assert reconciler._last_query_succeeded is True
    assert reconciler._last_prompt_delta_state["blocks"]["fear_greed"]["turns_omitted"] == 1
    assert "Snapshot Mode: WARM_DELTA" in stub.calls[-1]["kwargs"]["resume_prompt"]
    assert "Snapshot Mode: FULL_KEYFRAME" in stub.calls[-1]["prompt"]

    # 2. Warm resume fell back to detached full prompt (last_used_warm_resume=False) -> persists FULL state (0 omitted)
    stub.last_used_warm_resume = False
    with patch(
        "orchestratorOS.supervisor.autonomous_ai_reconciler.get_topic_thread_state",
        side_effect=[("cid-live", 2, True), ("cid-new", 0, False)],
    ):
        reconciler.synthesize_and_diagnose(
            sensors,
            logs,
            topic_key="auto_cryptotrade_executor",
            prev_prompt_delta_state=p1.next_state,
            urgent_escalation=False,
            extreme_escalation=False,
        )
    assert reconciler._last_query_succeeded is True
    assert reconciler._last_prompt_delta_state["blocks"]["fear_greed"]["turns_omitted"] == 0
    assert reconciler._last_prompt_delta_state["conversation_id"] == "cid-new"
    assert reconciler._last_prompt_delta_state["compacted_at_step"] == 0

    # 3. Query fails -> _last_query_succeeded is False and delta state is not mutated
    stub.response = None
    with patch(
        "orchestratorOS.supervisor.autonomous_ai_reconciler.get_topic_thread_state",
        return_value=("cid-live", 2, True),
    ):
        reconciler.synthesize_and_diagnose(
            sensors,
            logs,
            topic_key="auto_cryptotrade_executor",
            prev_prompt_delta_state=p1.next_state,
            urgent_escalation=False,
            extreme_escalation=False,
        )
    assert reconciler._last_query_succeeded is False
    assert reconciler._last_prompt_delta_state == {}


def test_gemini_client_warm_resume_flag_and_topic_thread_state(tmp_path) -> None:
    """Verify GeminiClient initializes/resets last_used_warm_resume and get_topic_thread_state handles empty/unknown topics."""
    client = GeminiClient(model="gemini-test", custom_runner=lambda p, m, t: '{"ok": true}')
    assert client.last_used_warm_resume is False
    client.last_used_warm_resume = True
    out = client.query_text("hello", cache_ttl_sec=0.0)
    assert out == '{"ok": true}'
    assert client.last_used_warm_resume is False

    reg_path = str(tmp_path / "missing_registry.json")
    assert get_topic_thread_state("unknown_topic_xyz", registry_path=reg_path) == ("", 0, False)
    assert get_topic_thread_state("auto_cryptotrade_executor", registry_path=reg_path) == ("", 0, False)


def test_wave1_tables_units_and_venue_specs_g2_elision() -> None:
    """Verify Wave 1 tables (Venue Specs G2, Open Orders, Equities), minutes>=120s, compact 3B USD, and true whale_long_pct."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    sensors = _make_rich_sensors(t0)
    sensors.prices_and_trends["BTCUSDC"].update(
        {
            "slope_90s": 0.00454,
            "slope_3_7m": -0.05656,
            "slope_15m": -0.016,
            "slope_2_5h": -0.04089,
            "growth_90s": 0.00454,
            "vel_90s_pct_min": -0.0378,
            "accel_90s_pct_min2": -0.0516,
            "kalman_vel": -0.00094,
            "kalman_vel_std": 0.031,
        }
    )
    sensors.source_ages_sec["sentiment_advisor"] = 540.0
    sensors.source_ages_sec["geopolitical_threat"] = 4200.0
    sensors.microstructure["BTCUSDC"]["whale_long_pct"] = 0.655
    sensors.portfolio_summary["open_limit_orders"] = [
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "order_type": "LIMIT",
            "order_id": "990011",
            "actor": "AI",
            "price": 59680.0,
            "qty": 0.015,
            "filled_qty": 0.0,
            "status": "NEW",
            "age": "2.5h",
            "dist_from_spot_pct": -1.35,
            "fill_prob_1h_pct": 25,
            "recommended_front_run_px": 60200.0,
        }
    ]
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)

    p1 = evaluate_prompt_delta_plan(
        sensors, asset_scope="crypto", now_ts=t0, is_warm_resume=False, conversation_id="w1-cid"
    )
    full_txt = build_supervisor_prompt(sensors, logs, compact_rules=False, delta_plan=p1)
    assert "sentiment_advisor=9.0m" in full_txt
    assert "geopolitical_threat=70.0m" in full_txt
    assert "whale_long_pct=65.5%" in full_txt
    assert "bid_depth=$450.0K, ask_depth=$380.0K" in full_txt
    assert "oi_usd=$1.50B" in full_txt
    assert "2W/0BE/1L" in full_txt
    assert "| Venue | Symbol | Tier | Fee/Leg (M/T bps) | MinNotional | MinQty | Step | Tick |" in full_txt
    assert "| binance | BTCUSDC | VIP0 |" in full_txt
    assert "| Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Note |" in full_txt
    assert "| binance | BTCUSDC | BUY | 59680.0 | 0.015 | 0 | 990011 | AI | 2.5h | -1.35% | 25% | front_run=60200.0 |" in full_txt
    assert "| Symbol | Px | 24h | 7d | Pos24h | ExtVsRTH | Close | Source (Age) |" in full_txt
    assert "slopes(" not in full_txt
    assert "g_coeff(" not in full_txt
    assert "`slopes`" not in full_txt
    assert "kinematics(vel_90s=-0.0378%/m, acc_90s=-0.0516%/m^2, kalman_z=-0.03)" in full_txt
    assert "kalman_vel=-0.00094" in full_txt
    assert audit_rendered_prompt_text(full_txt) == []

    # Warm Turn 2: Venue_Specs omitted (1/5), while open_limit_orders table (G0) stays visible
    t2 = t0 + 360.0
    sensors_t2 = _make_rich_sensors(t2, anchor_ts=t0)
    sensors_t2.portfolio_summary["open_limit_orders"] = sensors.portfolio_summary["open_limit_orders"]
    p2 = evaluate_prompt_delta_plan(
        sensors_t2,
        asset_scope="crypto",
        now_ts=t2,
        is_warm_resume=True,
        prev_state=p1.next_state,
        conversation_id="w1-cid",
    )
    assert p2.action_for("venue_specs") == "OMIT"
    warm_txt = build_supervisor_prompt(sensors_t2, logs, compact_rules=True, delta_plan=p2)
    assert "Venue_Specs(1/5)" in warm_txt
    assert "Venue Specs & Fees" not in warm_txt
    assert "| 990011 |" in warm_txt
    assert audit_rendered_prompt_text(warm_txt) == []

    # Invariant verifier catches non-positive precision inside the new Venue Specs table
    bad_tbl = full_txt.replace(
        "| hyperliquid | HYPEUSDC | T0 | 4.5 (1.5/4.5) | $10 | 0.01 | 0.01 | 0.001 |",
        "| hyperliquid | HYPEUSDC | T0 | 4.5 (1.5/4.5) | $10 | 0.01 | 0.01 | 0 |",
    )
    assert bad_tbl != full_txt
    assert any(v.code == "PROMPT_VENUE_PRECISION" for v in audit_rendered_prompt_text(bad_tbl))

    # Invariant verifier catches OB imbalance math corruption inside a unified multi-line symbol card
    bad_ob = full_txt.replace(
        "ob_imbalance=0.542 (bid_depth=$450.0K, ask_depth=$380.0K)",
        "ob_imbalance=0.95 (bid_depth=$450.0K, ask_depth=$380.0K)",
    )
    assert bad_ob != full_txt
    assert any(v.code == "PROMPT_OB_IMBALANCE_MATH" for v in audit_rendered_prompt_text(bad_ob))

