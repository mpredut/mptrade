"""Focused tests for pre-window attribution cohort separation and passive-offset LIMIT fill-rate TCA."""
from __future__ import annotations

import pytest

from providers.ai_attribution import evaluate_ai_attribution, format_ai_attribution_report
from providers.ai_survival_guard import (
    evaluate_ai_calibration_feedback,
    extract_limit_passive_offset_bps,
    format_ai_calibration_feedback_line,
    summarize_limit_passive_offset_fill_rates,
)


def test_attribution_separates_pre_window_legacy_carryover_from_in_window_cohort():
    """Pre-window unattributed legacy BUYs closed in-window must not pollute unknown model or in-window unplanned stats."""
    now_s = 1_795_700_000.0
    intents = [
        # Pre-window legacy BUY (8.5d ago, within 7d+7d hydration, model_used=None, no trade plan)
        {
            "intent_id": "leg_b1",
            "timestamp": now_s - 8.5 * 86400.0,
            "decision": "BUY",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "clamped_notional_usd": 1000.0,
            "execution_payload": {"orderId": "bn_leg_b1", "status": "FILLED", "executedQty": "0.015", "avg_fill_price": 68000.0},
        },
        # In-window SELL closing the pre-window legacy BUY at a loss (hold = 4.5d <= 7d max hold)
        {
            "intent_id": "leg_s1",
            "timestamp": now_s - 4.0 * 86400.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 940.0,
            "execution_payload": {"orderId": "bn_leg_s1", "status": "FILLED", "executedQty": "0.015", "avg_fill_price": 63920.0},
        },
        # In-window profitable BUY -> SELL on gemini-4-neon-medium (unplanned, in-window)
        {
            "intent_id": "win_b1",
            "timestamp": now_s - 2.0 * 86400.0,
            "decision": "BUY",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 500.0,
            "execution_payload": {"orderId": "hl_win_b1", "status": "FILLED", "executedQty": "10.0", "avg_fill_price": 50.0},
        },
        {
            "intent_id": "win_s1",
            "timestamp": now_s - 1.0 * 86400.0,
            "decision": "SELL",
            "symbol": "HYPEUSDC",
            "venue": "hyperliquid",
            "order_type": "LIMIT",
            "status": "EXECUTED",
            "model_used": "gemini-4-neon-medium",
            "clamped_notional_usd": 520.0,
            "execution_payload": {"orderId": "hl_win_s1", "status": "FILLED", "executedQty": "10.0", "avg_fill_price": 52.0},
        },
    ]

    rep = evaluate_ai_attribution(intents, now_ts=now_s, lookback_days=7.0, scope="crypto")

    # Full summary preserves both realized trades, while cohort split cleanly isolates in-window vs legacy carryover
    assert rep["ai_summary"]["closed_trades"] == 2
    assert rep["ai_summary_in_window"]["closed_trades"] == 1
    assert rep["ai_summary_in_window"]["realized_net_usd"] > 0.0
    assert rep["ai_summary_legacy_pre_window"]["closed_trades"] == 1
    assert rep["ai_summary_legacy_pre_window"]["realized_net_usd"] < 0.0

    # Pre-window unknown legacy close routes to legacy_pre_window rather than polluting in-window unknown model stats
    assert "legacy_pre_window" in rep["ai_by_model"]
    assert rep["ai_by_model"]["legacy_pre_window"]["scored"] == 1
    assert "unknown" not in rep["ai_by_model"]

    # In-window unplanned_performance excludes pre-window legacy carryover; legacy performance is reported separately
    tpc = rep["trade_plan_calibration"]
    assert tpc["unplanned_performance"]["scored"] == 1
    assert tpc["unplanned_performance"]["total_net_usd"] > 0.0
    assert tpc["legacy_pre_window_performance"]["scored"] == 1
    assert tpc["legacy_pre_window_performance"]["total_net_usd"] < 0.0

    txt = format_ai_attribution_report(rep)
    assert "AI Cohort Split: in_window=1" in txt
    assert "legacy_pre_window_exits=1" in txt
    assert "Legacy pre-window avg=" in txt
    assert "legacy_pre_window" in txt


def test_limit_passive_offset_fill_rate_tca_and_section_3c_feedback():
    """Verify symmetric BUY/SELL passive offset math, empirical fill-rate bands, per-venue unf_off, and objective Section 3C line."""
    # Symmetric passive offset math (>0 passive away from spot, <0 aggressive across spot)
    assert extract_limit_passive_offset_bps(
        {"order_type": "LIMIT", "decision": "BUY", "reference_price": 100.0, "limit_price": 99.0}
    ) == pytest.approx(100.0)
    assert extract_limit_passive_offset_bps(
        {"order_type": "LIMIT", "decision": "SELL", "reference_price": 100.0, "limit_price": 101.0}
    ) == pytest.approx(100.0)
    assert extract_limit_passive_offset_bps(
        {"order_type": "LIMIT", "decision": "BUY", "reference_price": 100.0, "limit_price": 100.1}
    ) == pytest.approx(-10.0)
    assert extract_limit_passive_offset_bps(
        {"order_type": "MARKET", "decision": "BUY", "reference_price": 100.0, "limit_price": 99.0}
    ) is None

    bands = summarize_limit_passive_offset_fill_rates([10.0, 35.0, 60.0], [20.0, 80.0, 120.0, 150.0])
    assert bands["le_40bps"] == {"label": "<=40bp", "filled": 2, "unfilled": 1, "total": 3, "fill_rate_pct": 66.7}
    assert bands["40_100bps"] == {"label": "40-100bp", "filled": 1, "unfilled": 1, "total": 2, "fill_rate_pct": 50.0}
    assert bands["gt_100bps"] == {"label": ">100bp", "filled": 0, "unfilled": 2, "total": 2, "fill_rate_pct": 0.0}

    now_s = 1_795_700_000.0
    intents = [
        # 2 filled tight LIMIT orders (<=40bp)
        {
            "intent_id": "f1",
            "timestamp": now_s - 7200.0,
            "decision": "SELL",
            "symbol": "BTCUSDC",
            "venue": "binance",
            "order_type": "LIMIT",
            "reference_price": 60000.0,
            "limit_price": 60090.0,  # +15bps
            "status": "EXECUTED",
            "clamped_notional_usd": 600.0,
            "execution_payload": {"orderId": "o_f1", "status": "FILLED", "executedQty": "0.01", "avg_fill_price": 60090.0},
        },
        {
            "intent_id": "f2",
            "timestamp": now_s - 6600.0,
            "decision": "BUY",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "reference_price": 200.0,
            "limit_price": 199.4,  # +30bps
            "status": "EXECUTED",
            "clamped_notional_usd": 400.0,
            "execution_payload": {"orderId": "o_f2", "status": "FILLED", "executedQty": "2.0", "avg_fill_price": 199.4},
        },
        # 3 canceled far passive Kraken LIMIT orders (>100bp) -> med_unfilled_offset_bps = +130bps
        {
            "intent_id": "u1",
            "timestamp": now_s - 5400.0,
            "decision": "SELL",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "reference_price": 200.0,
            "limit_price": 202.2,  # +110bps
            "status": "EXECUTED",
            "clamped_notional_usd": 400.0,
            "execution_payload": {"orderId": "o_u1", "status": "CANCELED", "executedQty": "0.0"},
        },
        {
            "intent_id": "u2",
            "timestamp": now_s - 4800.0,
            "decision": "SELL",
            "symbol": "TAOUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "reference_price": 200.0,
            "limit_price": 202.6,  # +130bps
            "status": "EXECUTED",
            "clamped_notional_usd": 400.0,
            "execution_payload": {"orderId": "o_u2", "status": "CANCELED", "executedQty": "0.0"},
        },
        {
            "intent_id": "u3",
            "timestamp": now_s - 4200.0,
            "decision": "BUY",
            "symbol": "HYPEUSD",
            "venue": "kraken",
            "order_type": "LIMIT",
            "reference_price": 50.0,
            "limit_price": 49.2,  # +160bps
            "status": "EXECUTED",
            "clamped_notional_usd": 400.0,
            "execution_payload": {"orderId": "o_u3", "status": "CANCELED", "executedQty": "0.0"},
        },
    ]

    cal = evaluate_ai_calibration_feedback(intents, now_ts=now_s, scope="crypto")
    assert cal["venue_tca"]["kraken"]["med_filled_offset_bps"] == pytest.approx(30.0)
    assert cal["venue_tca"]["kraken"]["med_unfilled_offset_bps"] == pytest.approx(130.0)
    assert cal["limit_offset_fill_rates"]["le_40bps"]["fill_rate_pct"] == pytest.approx(100.0)
    assert cal["limit_offset_fill_rates"]["gt_100bps"]["fill_rate_pct"] == pytest.approx(0.0)

    line = format_ai_calibration_feedback_line(cal)
    assert "kraken: LIM 1/4[25%]" in line
    assert "unfilled_off_bps=+130" in line
    assert "fill_by_offset(<=40bp:100%[2/2], >100bp:0%[0/3])" in line

    rep = evaluate_ai_attribution(intents, now_ts=now_s, scope="crypto")
    assert rep["execution_quality"]["by_venue"]["kraken"]["med_unfilled_offset_bps"] == pytest.approx(130.0)
    rendered = format_ai_attribution_report(rep)
    assert "LIMIT Offset Fill Rate: <=40bp: 100.0% (2/2), >100bp: 0.0% (0/3)" in rendered
    assert "UnfOff" in rendered
    assert "+130.0" in rendered

    # Calibration forwards caller now_ts deterministically into evaluate_filled_trade_plans (no wall-clock leak)
    import providers.ai_survival_guard as asg

    orig_eval_plans = asg.evaluate_filled_trade_plans
    seen_now: list[object] = []

    def _spy_eval_plans(*a, **kw):
        seen_now.append(kw.get("now_ts"))
        return orig_eval_plans(*a, **kw)

    asg.evaluate_filled_trade_plans = _spy_eval_plans  # type: ignore[assignment]
    try:
        evaluate_ai_calibration_feedback(intents, now_ts=now_s, scope="crypto")
    finally:
        asg.evaluate_filled_trade_plans = orig_eval_plans  # type: ignore[assignment]
    assert seen_now == [now_s]


def test_equity_bellwethers_strip_synthetic_perp_funding_and_whale_placeholders():
    """US equities in cross_market_context omit synthetic funding_annualized_pct=0.0 and whale_long_pct=0.5 without live CBOE OI."""
    from types import SimpleNamespace
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

    ctx = build_cross_market_context(
        {
            "NVDA_US_EQ": {"price": 151.3, "chg_24h_pct": -0.91},
            "SPCX_US_EQ": {"price": 128.95, "chg_24h_pct": 2.15},
        },
        {
            "NVDA_US_EQ": {
                "ob_imbalance": 0.923,
                "spread_bps": 6.67,
                "whale_long_pct": 0.5,
                "funding_rate": 0.0,
                "funding_annualized_pct": 0.0,
                "perp_spot_basis_bps": 0.0,
            },
            "SPCX_US_EQ": {
                "ob_imbalance": 0.909,
                "spread_bps": 0.78,
                "oi_usd": 5_000_000.0,  # Live CBOE options OI -> whale_long_pct IS real and preserved
                "whale_long_pct": 0.62,
                "funding_annualized_pct": 0.0,
            },
        },
    )
    nvda_bw = ctx["equity_bellwethers"]["NVDA_US_EQ"]
    assert "funding_annualized_pct" not in nvda_bw
    assert "funding_rate" not in nvda_bw
    assert "whale_long_pct" not in nvda_bw
    assert nvda_bw["ob_imbalance"] == pytest.approx(0.923)

    spcx_bw = ctx["equity_bellwethers"]["SPCX_US_EQ"]
    assert "funding_annualized_pct" not in spcx_bw
    assert spcx_bw["whale_long_pct"] == pytest.approx(0.62)

    # Prompt instructions reference unfilled_off_bps and fill_by_offset objectively alongside arr_bps/slip_bps
    for tk in ("auto_cryptotrade_executor", "auto_sharetrade_executor"):
        p_txt = build_supervisor_prompt(SimpleNamespace(), {}, topic_key=tk, max_usd=1000.0)
        assert "`arr_bps`/`slip_bps`" in p_txt
        assert "`unfilled_off_bps`" in p_txt and "`fill_by_offset`" in p_txt
        assert "`unf_off`" not in p_txt and "`lim_off`" not in p_txt
        if tk == "auto_sharetrade_executor":
            assert "negative = favorable price improvement" in p_txt

