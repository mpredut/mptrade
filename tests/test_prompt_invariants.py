"""Regression tests for unified AI intent LIFO windowing, TimedQuote, single-path portfolio normalization, and prompt invariants."""
from __future__ import annotations

import pytest

from botcore import TimedQuote, quote_spread_bps, resolve_aligned_quote_pair
from intelligence.sentiment.collectors.portfolio_collector import derive_runtime_portfolio_view
from intelligence.sentiment.evaluators.microstructure_evaluator import compute_cross_venue_arb_metrics
from providers.ai_attribution import evaluate_ai_attribution
from providers.ai_survival_guard import (
    evaluate_ai_calibration_feedback,
    evaluate_ai_decision_scorecard,
    match_filled_intent_rows_lifo_in_window,
)
from providers.intent_telemetry import summarize_active_ai_trade_plans
from intelligence.sentiment.evaluators.prompt_invariants_evaluator import (
    audit_rendered_prompt_text,
    audit_scorecard_temporal_invariants,
    audit_snapshot_invariants,
    evaluate_live_prompt_invariants,
)


def _filled_intent(
    iid: str,
    ts: float,
    side: str,
    symbol: str,
    venue: str,
    px: float,
    qty: float,
    *,
    model: str = "gemini-4-neon-high",
) -> dict:
    return {
        "intent_id": iid,
        "timestamp": ts,
        "decision": side,
        "symbol": symbol,
        "venue": venue,
        "order_type": "LIMIT",
        "status": "EXECUTED",
        "model_used": model,
        "clamped_notional_usd": round(px * qty, 2),
        "execution_payload": {
            "orderId": f"oid_{iid}",
            "status": "FILLED",
            "avg_fill_price": px,
            "executedQty": str(qty),
        },
    }


def test_scorecard_and_attribution_match_lifo_before_window_cutoff() -> None:
    """Pre-cutoff BUY entries must stay on the LIFO stack so in-window stop-loss SELLs are never inverted into phantom wins."""
    now_s = 1_800_000_000.0
    h = 3600.0
    # Real live failure pattern reproduced from PROD:
    # - BUY @ 66.15 (51.8h ago, BEFORE 48h cutoff) -> SELL @ 63.72 (45.5h ago, INSIDE 48h): real LOSS (-3.76%)
    # - BUY @ 62.35 (40.4h ago) -> SELL @ 63.55 (39.7h ago): real WIN (+1.84%)
    # Previously, dropping the 51.8h BUY pre-cutoff left SELL @ 63.72 orphaned and paired it with BUY @ 62.35 as a fake +2.06% WIN!
    intents = [
        _filled_intent("b1", now_s - 51.8 * h, "BUY", "HYPEUSDC", "hyperliquid", 66.15, 6.0),
        _filled_intent("s1", now_s - 45.5 * h, "SELL", "HYPEUSDC", "hyperliquid", 63.72, 6.0),
        _filled_intent("b2", now_s - 40.4 * h, "BUY", "HYPEUSDC", "hyperliquid", 62.35, 11.0),
        _filled_intent("s2", now_s - 39.7 * h, "SELL", "HYPEUSDC", "hyperliquid", 63.55, 11.0),
    ]

    sc = evaluate_ai_decision_scorecard(intents, now_ts=now_s, window_trades=5, lookback_hours=48.0, scope="crypto")
    assert sc["count"] == 2
    assert sc["total_outcomes"] == 2
    assert sc["losses"] == 1
    assert sc["wins"] == 1
    assert sc["unmatched_sells"] == 0
    # Net USD: real loss (~-$14.90) + real win (~+$12.65) < 0 (NOT inflated fake wins!)
    assert sc["net_usd"] < 0.0

    # Temporal stability: advancing clock 1h with zero new fills must keep overlapping closed pairs immutable.
    violations = audit_scorecard_temporal_invariants(
        intents, t_early=now_s - 1.0 * h, t_late=now_s, scope="crypto", lookback_hours=48.0
    )
    assert violations == []

    # Same pre-cutoff boundary tested on compute_ai_attribution (lookback_days=2.0 -> 48h) AND execution_audit actors.
    audit_recs = [
        {
            "ts": now_s - 51.8 * h,
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "order_id": "oid_b1",
            "client_order_id": "ai_exec_b1",
            "side": "BUY",
            "event": "filled",
            "filled_qty": 6.0,
            "price": 66.15,
            "cost_usd": 396.9,
        },
        {
            "ts": now_s - 45.5 * h,
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "order_id": "oid_s1",
            "client_order_id": "ai_exec_s1",
            "side": "SELL",
            "event": "filled",
            "filled_qty": 6.0,
            "price": 63.72,
            "cost_usd": 382.32,
        },
        {
            "ts": now_s - 40.4 * h,
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "order_id": "oid_b2",
            "client_order_id": "ai_exec_b2",
            "side": "BUY",
            "event": "filled",
            "filled_qty": 11.0,
            "price": 62.35,
            "cost_usd": 685.85,
        },
        {
            "ts": now_s - 39.7 * h,
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "order_id": "oid_s2",
            "client_order_id": "ai_exec_s2",
            "side": "SELL",
            "event": "filled",
            "filled_qty": 11.0,
            "price": 63.55,
            "cost_usd": 699.05,
        },
    ]
    attr = evaluate_ai_attribution(
        intents,
        now_ts=now_s,
        lookback_days=2.0,
        scope="crypto",
        audit_records=audit_recs,
    )
    # Only the 3 in-window intents count toward 48h fill volume, yet LIFO matches all 2 real round-trips accurately.
    assert attr["ai_filled_total"] == 3
    assert attr["ai_summary"]["closed_trades"] == 2
    assert attr["ai_summary"]["wins"] == 1
    assert attr["ai_summary"]["losses"] == 1
    assert attr["ai_summary"]["realized_net_usd"] < 0.0
    # Execution audit actor comparison must also match the pre-window BUY and report negative net_usd (not fake reload win).
    ai_audit = attr["execution_audit_ai_vs_bot"]["by_actor"]["AI"]
    assert attr["execution_audit_ai_vs_bot"]["total_filled_orders"] == 3
    assert ai_audit["matched_round_trips"] == 2
    assert ai_audit["realized_net_usd"] < 0.0


def test_lifo_window_helper_excludes_pre_window_open_buys_and_closed_exits() -> None:
    rows = [
        {"ts": 100.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "BUY", "px": 50000.0, "qty": 0.1, "usd": 5000.0},
        {"ts": 200.0, "inst_key": ("binance", "BTC", ("USD",)), "side": "SELL", "px": 51000.0, "qty": 0.1, "usd": 5100.0},
        {"ts": 250.0, "inst_key": ("binance", "ETH", ("USD",)), "side": "BUY", "px": 2000.0, "qty": 1.0, "usd": 2000.0},
        {"ts": 900.0, "inst_key": ("binance", "ETH", ("USD",)), "side": "SELL", "px": 1900.0, "qty": 1.0, "usd": 1900.0},
        {"ts": 150.0, "inst_key": ("binance", "TAO", ("USD",)), "side": "BUY", "px": 180.0, "qty": 2.0, "usd": 360.0},
    ]
    closed, open_b, open_s = match_filled_intent_rows_lifo_in_window(rows, cutoff_ts=500.0)
    assert len(closed) == 1
    assert closed[0][0] == "BUY->SELL"
    assert closed[0][1]["px"] == 2000.0 and closed[0][2]["px"] == 1900.0
    # Pre-window open BUY (TAO @ ts=150) is filtered out by post-match cutoff=500.
    assert open_b == []
    assert open_s == []


def test_timed_quote_alignment_and_arb_consistency() -> None:
    now_s = 1_700_000_100.0
    ob_a = TimedQuote.from_raw(61.635, now_s - 1.0, source="ob_mid")
    ob_b = TimedQuote.from_raw(61.640, now_s - 1.5, source="ob_mid")
    spot_a = TimedQuote.from_raw(61.636, source="spot")
    spot_b = TimedQuote.from_raw(61.641, source="spot")
    pair = resolve_aligned_quote_pair(ob_a, ob_b, spot_a, spot_b, now_ts=now_s)
    assert pair is not None
    assert pair[0] == ob_a and pair[1] == ob_b
    assert quote_spread_bps(pair[0], pair[1]) == pytest.approx(-0.8, abs=0.1)

    # Skewed OB timestamps with consistent live spot fall back to aligned spot quotes.
    stale_ob_b = TimedQuote.from_raw(61.640, now_s - 25.0, source="ob_mid")
    pair_fb = resolve_aligned_quote_pair(ob_a, stale_ob_b, spot_a, spot_b, now_ts=now_s)
    assert pair_fb == (spot_a, spot_b)

    # Skewed OB timestamps with a badly diverged (>1%) stale spot refuse to corrupt the spread with stale spot.
    bad_spot_b = TimedQuote.from_raw(55.0, source="spot")
    pair_guard = resolve_aligned_quote_pair(ob_a, stale_ob_b, spot_a, bad_spot_b, now_ts=now_s)
    assert pair_guard == (ob_a, stale_ob_b)

    # End-to-end mutual symmetry in compute_cross_venue_arb_metrics
    micro = {
        "HYPEUSDC": {"venue": "hyperliquid", "ob_mid_price": 61.635, "_ob_raw_ts": now_s - 1.0, "spread_bps": 0.2},
        "HYPEUSD": {"venue": "kraken", "ob_mid_price": 61.640, "_ob_raw_ts": now_s - 1.5, "spread_bps": 1.6},
    }
    pt = {"HYPEUSDC": {"price": 61.635}, "HYPEUSD": {"price": 61.640}}
    compute_cross_venue_arb_metrics("HYPEUSDC", micro["HYPEUSDC"], pt["HYPEUSDC"], micro, pt, now_ts=now_s)
    compute_cross_venue_arb_metrics("HYPEUSD", micro["HYPEUSD"], pt["HYPEUSD"], micro, pt, now_ts=now_s)
    assert micro["HYPEUSDC"]["cross_venue_arb_bps"] == pytest.approx(-micro["HYPEUSD"]["cross_venue_arb_bps"], abs=0.2)


def test_single_path_portfolio_normalizer_filters_ghosts_and_reconciles_quote_breakdown(tmp_path) -> None:
    """derive_runtime_portfolio_view enforces registry ghost filtering and quote breakdown reconciliation uniformly."""
    (tmp_path / "instruments.conf").write_text(
        "[KRAKEN_ANT]\n"
        "role.mt = no\n"
        "role.tradeall_fire = no\n"
        "role.kalman_primary = no\n"
        "role.trailing = no\n"
        "role.assetguardian = no\n"
        "role.archive = yes\n"
        "role.rtrade = no\n"
        "role.force_sell = no\n"
        "ai.tradable = no\n"
        "provider = kraken\n"
        "symbol = ANTxUSD\n"
        "base = ANTx\n"
        "quote = USD\n"
        "enabled = yes\n"
        "isolation = own_ledger\n"
        "market_hours = 24x7\n"
        "\n"
        "[KRAKEN_SPCXX]\n"
        "role.mt = no\n"
        "role.tradeall_fire = no\n"
        "role.kalman_primary = no\n"
        "role.trailing = no\n"
        "role.assetguardian = no\n"
        "role.archive = yes\n"
        "role.rtrade = no\n"
        "role.force_sell = no\n"
        "ai.tradable = yes\n"
        "provider = kraken\n"
        "symbol = SPCXxUSD\n"
        "base = SPCXx\n"
        "quote = USD\n"
        "enabled = yes\n"
        "isolation = own_ledger\n"
        "market_hours = 24x7\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir()
    now_s = 1_700_000_000.0
    positions = [
        {"venue": "kraken", "symbol": "ANTxUSD", "base": "ANTx", "quote": "USD", "free_base": 1.0, "free_quote": 1660.0, "price": 82.0},
        {
            "venue": "kraken",
            "symbol": "SPCXxUSD",
            "base": "SPCXx",
            "quote": "USD",
            "free_base": 12.0,
            "free_quote": 1660.02,
            "free_quote_breakdown": {"USD": 1100.0, "USDC": 10.0},  # stale >5% mismatch
            "price": 127.3,
        },
        {
            "venue": "kraken",
            "symbol": "SPCXxUSD",  # duplicate
            "base": "SPCXx",
            "quote": "USD",
            "free_base": 12.0,
            "free_quote": 1660.02,
            "price": 127.3,
        },
    ]
    derive_runtime_portfolio_view(
        [],
        positions,
        {"kraken:USD": {"venue": "kraken", "asset": "USD", "free": 1660.02}},
        {},
        {"SPCXxUSD": {"price": 127.3}},
        {},
        {},
        now_ts_oo=now_s,
        gateway_max_usd=2000.0,
        cache_dir=str(cache_dir),
        fresh_bal_cache_bd={("kraken", "USD"): {"USD": 1657.71, "USDC": 0.56, "USDG": 1.75}},
    )
    assert [p["symbol"] for p in positions] == ["SPCXxUSD"]
    assert positions[0]["free_quote_breakdown"] == {"USD": 1657.71, "USDC": 0.56, "USDG": 1.75}

    # If no fresh breakdown matches updated free_quote (e.g., >5% stale), stale breakdown is stripped instead of rendered.
    positions_stale = [
        {
            "venue": "kraken",
            "symbol": "SPCXxUSD",
            "base": "SPCXx",
            "quote": "USD",
            "free_base": 12.0,
            "free_quote": 500.0,
            "free_quote_breakdown": {"USD": 1657.71},
            "price": 127.3,
        }
    ]
    derive_runtime_portfolio_view(
        [],
        positions_stale,
        {"kraken:USD": {"venue": "kraken", "asset": "USD", "free": 500.0}},
        {},
        {"SPCXxUSD": {"price": 127.3}},
        {},
        {},
        now_ts_oo=now_s,
        gateway_max_usd=2000.0,
        cache_dir=str(cache_dir),
    )
    assert "free_quote_breakdown" not in positions_stale[0]


def test_prompt_and_snapshot_invariant_verifier_passes_clean_and_catches_bugs() -> None:
    clean_snap = {
        "prices_and_trends": {
            "BTCUSDC": {
                "price": 60000.0,
                "low_24h": 58000.0,
                "high_24h": 62000.0,
                "pos_24h_pct": 50.0,
                "major_high": 62000.0,
                "major_low": 58000.0,
            }
        },
        "microstructure": {
            "HYPEUSDC": {"cross_venue_peer": "HYPEUSD", "cross_venue_arb_bps": -0.8, "ob_mid_price": 61.635, "bid_wall_px": 61.5, "ask_wall_px": 61.8},
            "HYPEUSD": {"cross_venue_peer": "HYPEUSDC", "cross_venue_arb_bps": 0.8, "ob_mid_price": 61.640},
        },
        "portfolio_summary": {
            "portfolio_risk": {
                "crypto_nav_usd": 10000.0,
                "invested_usd": 1500.0,
                "free_cash_usd": 7600.0,
                "locked_quote_usd": 900.0,
                "venue_nav_usd": {"binance": 6000.0, "kraken": 4000.0},
            },
            "tradable_positions": [
                {"venue": "kraken", "symbol": "SPCXxUSD", "free_quote": 1660.02, "free_quote_breakdown": {"USD": 1657.71, "USDC": 0.56, "USDG": 1.75}}
            ],
        },
    }
    assert audit_snapshot_invariants(clean_snap, non_tradable_symbols={"ANTxUSD"}) == []

    # Intentionally inject multiple known bug classes and assert every single one is caught.
    broken_snap = {
        "prices_and_trends": {
            "BTCUSDC": {
                "price": 63000.0,
                "low_24h": 58000.0,
                "high_24h": 62000.0,
                "pos_24h_pct": 50.0,  # should be 100%
                "major_high": 62000.0,  # live price > major_high
            }
        },
        "microstructure": {
            "HYPEUSDC": {"cross_venue_peer": "HYPEUSD", "cross_venue_arb_bps": -15.0, "ob_mid_price": 61.6, "bid_wall_px": 63.0},
            "HYPEUSD": {"cross_venue_peer": "HYPEUSDC", "cross_venue_arb_bps": 0.8},
        },
        "portfolio_summary": {
            "portfolio_risk": {
                "crypto_nav_usd": 10000.0,
                "invested_usd": 1500.0,
                "free_cash_usd": 7000.0,
                "locked_quote_usd": 900.0,
                "venue_nav_usd": {"binance": 5000.0, "kraken": 4000.0},
            },
            "tradable_positions": [
                {"venue": "kraken", "symbol": "ANTxUSD", "free_quote": 1660.0, "free_quote_breakdown": {"USD": 1000.0}},
            ],
        },
    }
    codes = {v.code for v in audit_snapshot_invariants(broken_snap, non_tradable_symbols={"ANTxUSD"})}
    assert {
        "NAV_CONSERVATION",
        "NO_NON_TRADABLE_GHOSTS",
        "FREE_QUOTE_BREAKDOWN_CONSISTENCY",
        "CROSS_VENUE_ARB_SYMMETRY",
        "RANGE_POS_MATH",
        "SWING_WAVE_EXTREMES",
        "ORDERBOOK_WALL_ORDERING",
    }.issubset(codes)

    # Rendered prompt text audit catches prompt-level math regressions directly.
    good_prompt = (
        "  - BTCUSDC: Price=60000.0 | 24h_range=[58000.0..62000.0] (pos=50.0%)\n"
        "  - HYPEUSDC: arb_vs_HYPEUSD=-0.8bps\n"
        "  - HYPEUSD: arb_vs_HYPEUSDC=+0.8bps\n"
        "  * Portfolio Risk Engine: crypto_nav=$10,000.00, invested=$1,500.00, free_cash=$7,600.00 (76.0%), locked_quote=$900.00 (9.0%), venues(binance=$6,000.00, kraken=$4,000.00)\n"
        "    - [kraken] SPCXxUSD: free_SPCXx=12.0 ($1500 USD), free_USD=$1660.02 (USD=$1,657.71, USDC=$0.56, USDG=$1.75), limits(min_notional=$1)\n"
    )
    assert audit_rendered_prompt_text(good_prompt, non_tradable_symbols={"ANTxUSD"}) == []

    bad_prompt = (
        "  - BTCUSDC: Price=60000.0 | 24h_range=[58000.0..62000.0] (pos=12.0%)\n"
        "  - HYPEUSDC: arb_vs_HYPEUSD=-12.0bps\n"
        "  - HYPEUSD: arb_vs_HYPEUSDC=+0.8bps\n"
        "  * Portfolio Risk Engine: crypto_nav=$10,000.00, invested=$1,500.00, free_cash=$7,000.00 (70.0%), locked_quote=$900.00 (9.0%), venues(binance=$5,000.00, kraken=$4,000.00)\n"
        "    - [kraken] ANTxUSD: free_ANTx=1.0 ($82 USD), free_USD=$1660.02 (USD=$900.00), limits(min_notional=$1)\n"
    )
    bad_codes = {v.code for v in audit_rendered_prompt_text(bad_prompt, non_tradable_symbols={"ANTxUSD"})}
    assert {
        "PROMPT_NAV_CONSERVATION",
        "PROMPT_NO_NON_TRADABLE_GHOSTS",
        "PROMPT_FREE_QUOTE_BREAKDOWN",
        "PROMPT_RANGE_POS_MATH",
        "PROMPT_CROSS_VENUE_ARB_SYMMETRY",
    }.issubset(bad_codes)

    # Extended rendered prompt invariants catch vol_profile order/deltas, session_vwap delta,
    # OB imbalance/walls, T212 static precision regressions, and sub-gateway-min caps.
    ext_bad_prompt = (
        "  * NVDA_US_EQ: Price=$150.00 | vol_profile(POC=160.0[+5.00%], VAL=145.0[+3.45%], VAH=155.0[-3.23%], va_pos=50.0%, vp_source=REAL_VOL) "
        "| session_vwap=$140.00 (+1.00%, z=+1.0, prev_rth)\n"
        "  * BTCUSDC: ob_imbalance=0.900 (bid_depth=$50000.0, ask_depth=$50000.0), bid_wall=61000.0 ($10000.0, dist=+1.67%), ask_wall=59000.0 ($10000.0, dist=-1.67%), ob_mid=60000.0\n"
        "    - [t212] RGNT_US_EQ: free_RGNT_US_EQ=3.93 ($3.54 USD), free_USD=$608.63, px=0.9, limits(min_notional=$1, min_qty=0.001, step=0.001, tick=0.01, max_buy=$605.59, max_sell=$3.54)\n"
    )
    ext_codes = {v.code for v in audit_rendered_prompt_text(ext_bad_prompt, gateway_min_usd=10.0)}
    assert {
        "PROMPT_VOL_PROFILE_ORDER",
        "PROMPT_VOL_PROFILE_DELTA",
        "PROMPT_SESSION_VWAP_DELTA",
        "PROMPT_OB_IMBALANCE_MATH",
        "PROMPT_OB_WALL_ORDERING",
        "PROMPT_VENUE_PRECISION",
        "PROMPT_SUB_GATEWAY_MIN_CAP",
    }.issubset(ext_codes)


def test_frozen_resolved_plan_status_persists_write_once_and_survives_4h_bar_rollover(tmp_path) -> None:
    """Once a filled plan hits TARGET_HIT on 5m bars and is frozen in durable state, coarse 4h bars never flip it to INVALIDATED."""
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway

    ws = tmp_path / "ws"
    (ws / "cachedb").mkdir(parents=True)
    (ws / "order_guard.conf").write_text(
        "autonomous_trading_mode = shadow\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n"
        "autonomous_trading_notify = 0\n",
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway(workspace_dir=str(ws), cache_dir="cachedb", mode="shadow")

    t0 = 1_800_000_000.0
    sell_intent = {
        **_filled_intent("sell_plan_1", t0, "SELL", "ARBUSDC", "binance", 0.1100, 1000.0),
        "target_px": 0.1080,
        "invalidation_px": 0.1115,
    }
    assert gw.save_all_intents([sell_intent]) is True

    # 1. Within 24h on fine 5m bars ([ts, close, vol, high, low]):
    #    bar 1 hits target (low=0.1075), bar 2 later rebounds past invalidation (high=0.1122).
    fine_bars = [
        [t0 + 300.0, 0.1082, 100.0, 0.1102, 0.1075],
        [t0 + 600.0, 0.1118, 100.0, 0.1122, 0.1080],
    ]
    newly_resolved: dict[str, str] = {}
    cal_fine = evaluate_ai_calibration_feedback(
        gw.load_all_intents(),
        now_ts=t0 + 3600.0,
        live_prices={"binance:ARBUSDC": 0.1118},
        price_bars_by_symbol={"ARBUSDC": fine_bars},
        lookback_days=7.0,
        scope="crypto",
        newly_resolved_out=newly_resolved,
    )
    assert cal_fine["by_side"]["SELL"]["TARGET_HIT"] == 1
    assert cal_fine["by_side"]["SELL"]["INVALIDATED"] == 0
    assert newly_resolved == {"sell_plan_1": "TARGET_HIT"}

    # 2. Persist write-once via AutonomousIntentGateway; subsequent conflicting write must be ignored.
    assert gw.persist_resolved_trade_plan_statuses(newly_resolved, now_ts=t0 + 3600.0) == 1
    assert gw.persist_resolved_trade_plan_statuses({"sell_plan_1": "INVALIDATED"}, now_ts=t0 + 7200.0) == 0

    persisted = gw.load_all_intents()
    assert persisted[0]["resolved_plan_status"] == "TARGET_HIT"
    assert persisted[0]["resolved_plan_ts"] == pytest.approx(t0 + 3600.0)

    # 3. Past 24h rollover into coarse 4h bar ([ts, close, vol, high, low]) spanning BOTH target and invalidation:
    #    verify unfrozen intent WOULD flip to INVALIDATED, while frozen persisted intent stays TARGET_HIT across all evaluators!
    coarse_4h_bars = [
        [t0 + 14400.0, 0.1120, 5000.0, 0.1125, 0.1075],
    ]
    unfrozen_cal = evaluate_ai_calibration_feedback(
        [sell_intent],
        now_ts=t0 + 90000.0,
        live_prices={"binance:ARBUSDC": 0.1120},
        price_bars_by_symbol={"ARBUSDC": coarse_4h_bars},
        lookback_days=7.0,
        scope="crypto",
    )
    assert unfrozen_cal["by_side"]["SELL"]["INVALIDATED"] == 1

    recheck_resolved: dict[str, str] = {}
    cal_4h = evaluate_ai_calibration_feedback(
        persisted,
        now_ts=t0 + 90000.0,
        live_prices={"binance:ARBUSDC": 0.1120},
        price_bars_by_symbol={"ARBUSDC": coarse_4h_bars},
        lookback_days=7.0,
        scope="crypto",
        newly_resolved_out=recheck_resolved,
    )
    assert cal_4h["by_side"]["SELL"]["TARGET_HIT"] == 1
    assert cal_4h["by_side"]["SELL"]["INVALIDATED"] == 0
    assert recheck_resolved == {}

    attr_4h = evaluate_ai_attribution(
        persisted,
        now_ts=t0 + 90000.0,
        live_prices={"binance:ARBUSDC": 0.1120},
        price_bars_by_symbol={"ARBUSDC": coarse_4h_bars},
        lookback_days=7.0,
        scope="crypto",
    )
    assert attr_4h["trade_plan_calibration"]["status_counts"]["TARGET_HIT"] == 1
    assert attr_4h["trade_plan_calibration"]["status_counts"]["INVALIDATED"] == 0

    active_plans = summarize_active_ai_trade_plans(
        persisted,
        "crypto",
        t0 + 3600.0,
        lambda it, default=0.0: it.get("clamped_notional_usd", default),
        prices_and_trends={"ARBUSDC": {"price": 0.1120}},
        price_bars_by_symbol={"ARBUSDC": coarse_4h_bars},
    )
    assert active_plans and active_plans[0]["status"] == "TARGET_HIT"

    # 4. CLOSED_EARLY without price bars is NOT prematurely frozen, whereas with bars it IS frozen.
    early_intents = [
        {
            **_filled_intent("buy_early_1", t0, "BUY", "BTCUSDC", "binance", 60000.0, 0.01),
            "target_px": 62000.0,
            "invalidation_px": 58000.0,
        },
        _filled_intent("sell_early_1", t0 + 1800.0, "SELL", "BTCUSDC", "binance", 60200.0, 0.01),
    ]
    res_no_bars: dict[str, str] = {}
    evaluate_ai_calibration_feedback(
        early_intents,
        now_ts=t0 + 3600.0,
        live_prices={"binance:BTCUSDC": 60200.0},
        price_bars_by_symbol=None,
        newly_resolved_out=res_no_bars,
    )
    assert res_no_bars == {}

    res_with_bars: dict[str, str] = {}
    evaluate_ai_calibration_feedback(
        early_intents,
        now_ts=t0 + 3600.0,
        live_prices={"binance:BTCUSDC": 60200.0},
        price_bars_by_symbol={"BTCUSDC": [[t0 + 900.0, 60150.0, 50.0, 60300.0, 59900.0]]},
        newly_resolved_out=res_with_bars,
    )
    assert res_with_bars == {"buy_early_1": "CLOSED_EARLY"}


def test_lifo_window_bounds_pre_window_hydration_against_ancient_ghost_buys() -> None:
    """Weeks-old unmatched AI BUYs (sold by BOT/MANUAL on shared wallet) must not poison recent SELLs as fake -25% losses."""
    now_s = 1_800_000_000.0
    d = 86400.0
    intents = [
        # 25-day-old AI BUY that was sold weeks ago by a deterministic bot (absent from AI intent log)
        _filled_intent("ancient_b", now_s - 25.0 * d, "BUY", "HYPEUSDC", "hyperliquid", 90.0, 10.0),
        # Recent AI SELL of bot-bought inventory at $65 (should be standalone unmatched sell, NOT a -27.8% loss!)
        _filled_intent("recent_s", now_s - 1.0 * d, "SELL", "HYPEUSDC", "hyperliquid", 65.0, 10.0),
        # One real recent winning round-trip so scorecard is non-empty and reports unmatched_sells=1
        _filled_intent("win_b", now_s - 6.0 * 3600.0, "BUY", "BTCUSDC", "binance", 60000.0, 0.01),
        _filled_intent("win_s", now_s - 2.0 * 3600.0, "SELL", "BTCUSDC", "binance", 61200.0, 0.01),
    ]
    sc = evaluate_ai_decision_scorecard(intents, now_ts=now_s, window_trades=5, lookback_hours=48.0, scope="crypto")
    assert sc["count"] == 1
    assert sc["wins"] == 1
    assert sc["losses"] == 0
    assert sc["unmatched_sells"] == 1
    assert sc["net_usd"] > 0.0

    attr = evaluate_ai_attribution(intents, now_ts=now_s, lookback_days=7.0, scope="crypto")
    assert attr["ai_summary"]["closed_trades"] == 1
    assert attr["ai_summary"]["wins"] == 1
    assert attr["ai_summary"]["losses"] == 0
    assert attr["ai_summary"]["realized_net_usd"] > 0.0


def test_cycle11_12_acute_geo_escalation_gateway_min_snapshot_and_equity_ob_precision() -> None:
    """Verify Cycle 11 & 12 fixes:
    1. New acute unassessed geo headline escalates Delta-Gate to high-effort & urgent routing.
    2. BUY active trade plan clears symmetrically when effective_max_sell_usd == 0.0 (unsellable dust).
    3. derive_runtime_portfolio_view & PositionCostBasisEnrichmentEvaluator propagate gateway_min_usd
       and preserve T212 static precision.
    4. align_equity_orderbook_to_spot caps sub-dollar US equity OB quotes to <=4 decimal places.
    """
    from types import SimpleNamespace
    from intelligence.sentiment.evaluators.microstructure_evaluator import align_equity_orderbook_to_spot
    from intelligence.sentiment.evaluators.portfolio_evaluator import enrich_positions_with_orders_and_cycles
    from orchestratorOS.supervisor.crypto_executor_delta_gate import (
        evaluate_crypto_executor_delta_gate,
        is_urgent_delta_gate_wakeup,
    )

    t0 = 1_800_000_000.0
    acute_note = "BREAKING: Emergency missile strike hits Strait of Hormuz oil tanker"
    sensors_acute = SimpleNamespace(
        prices_and_trends={"BTCUSDC": {"price": 60500.0}},
        microstructure={},
        portfolio_summary={"tradable_positions": [], "open_orders": [], "recent_orders": {}},
        sentiment_advisor={"market_bias": "NEUTRAL", "recommended_action": "HOLD"},
        geopolitical_threat={
            "threat_level": "NORMAL",
            "recommended_brake": "NONE",
            "risk_score": 0.15,
            "live_unassessed_high_sev_note": acute_note,
        },
        source_timestamps={},
    )
    prev_hold = {
        "ts": t0 - 300.0,
        "model_used": "gemini-4-neon-medium",
        "market_trading_decisions": [{"decision": "HOLD", "symbol": "BTCUSDC"}],
        "delta_gate_meta": {
            "eval_prices": {"BTCUSDC": 60500.0},
            "macro_ts": t0 - 600.0,
            "macro_verdict": "NEUTRAL||HOLD|",
            "geo_ts": t0 - 1200.0,
            "geo_verdict": "NORMAL|NONE",
            "latest_fill_ts": 0.0,
            "open_order_ids": [],
            "order_attention_flags": {},
            "plan_states": {},
        },
    }
    skip_a, rsn_a, high_a, meta_a = evaluate_crypto_executor_delta_gate(sensors_acute, prev_hold, now_ts=t0)
    assert skip_a is False
    assert rsn_a == "geo_unassessed_acute_headline"
    assert high_a is True
    assert is_urgent_delta_gate_wakeup(rsn_a) is True
    assert is_urgent_delta_gate_wakeup("max_idle_elapsed", sensors=sensors_acute, prev_eval=prev_hold) is True
    prev_ack = {**prev_hold, "ts": t0, "delta_gate_meta": meta_a}
    assert is_urgent_delta_gate_wakeup("max_idle_elapsed", sensors=sensors_acute, prev_eval=prev_ack) is False

    # 2. BUY plan clears when effective_max_sell_usd == 0.0 (e.g. $7 unsellable dust < $10 gateway min)
    buy_dust_intent = {
        **_filled_intent("buy_rgnt_1", t0 - 3600.0, "BUY", "RGNT_US_EQ", "t212", 0.90, 8.0),
        "target_px": 1.10,
        "invalidation_px": 0.80,
    }
    plans_dust = summarize_active_ai_trade_plans(
        [buy_dust_intent],
        "equities",
        t0,
        lambda it, default=0.0: it.get("clamped_notional_usd", default),
        tradable_positions=[
            {
                "venue": "t212",
                "symbol": "RGNT_US_EQ",
                "value_usd": 7.20,
                "effective_max_buy_usd": 500.0,
                "effective_max_sell_usd": 0.0,
            }
        ],
        prices_and_trends={"RGNT_US_EQ": {"price": 0.90}},
    )
    assert plans_dust == []

    # 3. PositionCostBasisEnrichmentEvaluator propagates gateway_min_usd and enforces T212 static precision
    pos_list = [
        {
            "venue": "t212",
            "symbol": "RGNT_US_EQ",
            "price": 0.90,
            "free_base": 3.93,
            "free_quote": 500.0,
            "value_usd": 3.54,
            "min_qty": 0.001,
            "step_size": 0.001,
            "tick_size": 0.001,
        }
    ]
    enrich_positions_with_orders_and_cycles(
        pos_list,
        {"RGNT_US_EQ": {"price": 0.90}},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        gateway_max_usd=2000.0,
        gateway_min_usd=10.0,
        now_ts=t0,
    )
    assert pos_list[0]["effective_max_sell_usd"] == 0.0
    assert (pos_list[0]["min_qty"], pos_list[0]["step_size"], pos_list[0]["tick_size"]) == (0.01, 0.01, 0.01)

    # 4. align_equity_orderbook_to_spot caps sub-dollar US equities to <=4 decimals (no 0.895001)
    rgnt_ob = {
        "best_bid": 0.8951,
        "best_ask": 0.9051,
        "ob_mid_price": 0.9001,
        "bid_wall_px": 0.8951,
        "ask_wall_px": 0.9051,
        "spread_bps": 111.1,
    }
    assert align_equity_orderbook_to_spot(rgnt_ob, 0.90) is True
    for k in ("best_bid", "best_ask", "ob_mid_price", "bid_wall_px", "ask_wall_px"):
        frac = f"{rgnt_ob[k]:.8f}".rstrip("0").partition(".")[2]
        assert len(frac) <= 4, f"{k}={rgnt_ob[k]} has >4 decimals"


def test_cycle13_closed_equity_pruning_open_order_wall_notes_and_plain_venue_specs() -> None:
    """Verify Cycle 13 fixes:
    1. Closed-market equities prune bottom_structure_6h, kalman_vel_z, and elapsed 3h/6h/12h horizons
       in both EquityRthSessionPolicy and CrossMarketContextEvaluator while preserving 24h/7d/slope_2_5h
       (and preserving 3h/6h/12h when market closed <180m ago).
    2. Open-limit-orders table surfaces blocking wall_queue_pos + wall details in Note when front_run is absent
       (symmetrically for BUY and SELL) without cluttering FRONT_OF_* orders.
    3. Venue Specs table and inline limits emit plain decimals (no scientific notation 1e-05/1e-08),
       and audit_snapshot_invariants / audit_rendered_prompt_text validate all rows & stale closed-equity leaks.
    """
    from intelligence.sentiment.evaluators.asset_session_policy import EquityRthSessionPolicy
    from intelligence.sentiment.evaluators.tca_risk_evaluator import build_cross_market_context
    from orchestratorOS.supervisor.prompt_sections import (
        _format_open_limit_orders_table,
        _format_tradable_position_line,
        _format_venue_specs_table,
    )

    # 1a. Closed holiday/overnight equity (quote_age_min=751.2m = 12.5h) prunes intraday, bottom_6h, kalman_vel_z, 3h/6h/12h.
    pol = EquityRthSessionPolicy()
    holiday_entry = {
        "price": 0.90,
        "quote_age_min": 751.2,
        "chg_15m_pct": 0.1,
        "chg_1h_pct": -0.2,
        "chg_3h_pct": -0.25,
        "chg_6h_pct": 3.69,
        "chg_12h_pct": 12.5,
        "chg_24h_pct": 12.5,
        "chg_7d_pct": -1.1,
        "slope_2_5h": -0.00025,
        "kalman_vel": -0.12,
        "kalman_vel_std": 0.04,
        "kalman_vel_z": -2.78,
        "kalman_trend": -1,
        "bottom_structure_6h": "FRESH_6H_LOW_TEST",
        "prior_6h_low_px": 0.9023,
        "recent_1h_low_px": 0.90,
        "higher_low_vs_prior_pct": 0.5,
        "reclaim_above_prior_low_pct": 0.2,
    }
    holiday_m24 = dict(holiday_entry)
    pol.prune_stale_short_metrics(holiday_entry, holiday_m24, [], 1_800_000_000.0, post_rth_series_replaced=True)
    for pruned_k in (
        "chg_15m_pct", "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct",
        "kalman_vel", "kalman_vel_std", "kalman_vel_z", "kalman_trend",
        "bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px",
        "higher_low_vs_prior_pct", "reclaim_above_prior_low_pct",
    ):
        assert pruned_k not in holiday_entry, f"expected {pruned_k} pruned from entry"
        assert pruned_k not in holiday_m24, f"expected {pruned_k} pruned from m24"
    assert holiday_entry["chg_24h_pct"] == 12.5
    assert holiday_entry["chg_7d_pct"] == -1.1
    assert holiday_entry["slope_2_5h"] == pytest.approx(-0.00025)

    # 1b. Recently closed equity (45m ago) prunes 15m/1h/bottom_6h/kalman_vel_z but keeps 3h/6h/12h RTH changes.
    recent_closed = {
        "price": 151.3,
        "quote_age_min": 45.0,
        "chg_1h_pct": 0.4,
        "chg_3h_pct": 1.87,
        "chg_6h_pct": 2.33,
        "chg_12h_pct": 3.24,
        "kalman_vel_z": -0.69,
        "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
        "recent_1h_low_px": 150.0,
        "prior_6h_low_px": 149.0,
    }
    pol.prune_stale_short_metrics(recent_closed, {}, [], 1_800_000_000.0, post_rth_series_replaced=True)
    assert "chg_1h_pct" not in recent_closed and "kalman_vel_z" not in recent_closed and "bottom_structure_6h" not in recent_closed
    assert recent_closed["chg_3h_pct"] == 1.87 and recent_closed["chg_6h_pct"] == 2.33 and recent_closed["chg_12h_pct"] == 3.24

    # 1c. Cross-market equity bellwethers also prune elapsed 3h/6h/12h & bottom_structure_6h.
    cmc = build_cross_market_context(
        {
            "ANT_US_EQ": {
                "price": 84.29,
                "quote_age_min": 751.2,
                "chg_3h_pct": 1.97,
                "chg_6h_pct": -0.49,
                "chg_12h_pct": 7.8,
                "chg_24h_pct": 4.59,
                "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
                "prior_6h_low_px": 81.955,
                "recent_1h_low_px": 84.1,
            }
        },
        {},
    )
    ant_bw = cmc["equity_bellwethers"]["ANT_US_EQ"]
    assert "chg_3h_pct" not in ant_bw and "chg_6h_pct" not in ant_bw and "chg_12h_pct" not in ant_bw
    assert "bottom_structure_6h" not in ant_bw and "recent_1h_low_px" not in ant_bw
    assert ant_bw["chg_24h_pct"] == 4.59

    # 1d. Snapshot invariant catches leaked closed-equity stale keys in both prices_and_trends and equity_bellwethers.
    stale_snap = {
        "prices_and_trends": {
            "RGNT_US_EQ": {"price": 0.90, "quote_age_min": 751.2, "chg_6h_pct": 3.69, "bottom_structure_6h": "FRESH_6H_LOW_TEST"}
        },
        "portfolio_summary": {
            "cross_market_context": {
                "equity_bellwethers": {"ANT_US_EQ": {"price": 84.29, "quote_age_min": 751.2, "chg_6h_pct": -0.49}}
            }
        },
    }
    stale_vs = [v for v in audit_snapshot_invariants(stale_snap) if v.code == "CLOSED_EQUITY_STALE_METRICS"]
    assert len(stale_vs) == 2

    # 2. PortfolioPromptViewEvaluator preserves wall_queue_pos and open limit orders table surfaces blocking
    #    wall_queue_pos + wall px/usd symmetrically when front_run is absent.
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_portfolio_prompt_view

    raw_oo_list = [
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "price": 59680.0,
            "qty": 0.01508,
            "filled_qty": 0.0,
            "order_id": "10976647800",
            "actor": "AI",
            "age": "13.3h",
            "dist_from_spot_pct": -1.81,
            "fill_prob_1h_pct": 14.1,
            "wall_queue_pos": "DEEP_PASSIVE_BID",
            "bid_wall_ahead_px": 60711.36,
            "bid_wall_ahead_usd": 440480.0,
        },
        {
            "venue": "binance",
            "symbol": "ARBUSDC",
            "side": "BUY",
            "price": 0.1125,
            "qty": 2666.6,
            "filled_qty": 0.0,
            "order_id": "568650889",
            "actor": "AI",
            "age": "8.8h",
            "dist_from_spot_pct": -0.18,
            "fill_prob_1h_pct": 98.4,
            "wall_queue_pos": "FRONT_OF_BID_WALL",
        },
        {
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "side": "SELL",
            "price": 63.0,
            "qty": 10.0,
            "filled_qty": 0.0,
            "order_id": "999",
            "actor": "AI",
            "age": "1.0h",
            "dist_from_spot_pct": 2.5,
            "fill_prob_1h_pct": 12.0,
            "wall_queue_pos": "FAR_ABOVE_ASK_WALL",
            "ask_wall_ahead_px": 61.5,
            "ask_wall_ahead_usd": 1250000.0,
        },
    ]
    pv_oo = evaluate_portfolio_prompt_view({"open_limit_orders": raw_oo_list}, asset_scope="crypto")
    oo_lines = "\n".join(_format_open_limit_orders_table(pv_oo["clean_open_limit_orders"]))
    assert "DEEP_PASSIVE_BID(wall=60711.36/$440.5K)" in oo_lines
    assert "FAR_ABOVE_ASK_WALL(wall=61.5/$1.25M)" in oo_lines
    assert "FRONT_OF_BID_WALL" not in oo_lines

    # 3. Venue Specs table & inline limits emit plain decimals (0.00001 / 0.00000001) and audit catches sci-notation too.
    vspec_lines = "\n".join(
        _format_venue_specs_table(
            [
                {
                    "venue": "binance",
                    "disp_symbol": "BTCUSDC",
                    "fee_tier": "STANDARD",
                    "maker_bps": 10.0,
                    "taker_bps": 10.0,
                    "min_notional_usd": 5.0,
                    "limit_items": [("min_qty=", 1e-05), ("step=", 1e-05), ("tick=", 0.01)],
                },
                {
                    "venue": "kraken",
                    "disp_symbol": "HYPEUSD",
                    "fee_tier": "TIER_0",
                    "maker_bps": 22.0,
                    "taker_bps": 37.0,
                    "min_notional_usd": 1.0,
                    "limit_items": [("min_qty=", 0.06), ("step=", 1e-08), ("tick=", 0.01)],
                },
            ]
        )
    )
    assert "1e-05" not in vspec_lines and "1e-08" not in vspec_lines
    assert "| 0.00001 | 0.00001 | 0.01 |" in vspec_lines
    assert "| 0.06 | 0.00000001 | 0.01 |" in vspec_lines
    pos_line = _format_tradable_position_line(
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "price": 60000.0,
            "free_base": 0.01,
            "value_usd": 600.0,
            "free_quote": 1000.0,
            "min_qty": 1e-05,
            "step_size": 1e-05,
            "tick_size": 0.01,
        },
        max_usd=2000.0,
        min_usd=10.0,
        include_static_specs=True,
    )
    assert "min_qty=0.00001, step=0.00001" in pos_line and "1e-05" not in pos_line
    sci_bad_t212 = "    | t212 | RGNT_US_EQ | T212_ZERO_COMM | 0 (0/0) | $1 | 1e-05 | 1e-05 | 0.01 |"
    sci_codes = {v.code for v in audit_rendered_prompt_text(sci_bad_t212)}
    assert "PROMPT_VENUE_PRECISION" in sci_codes


def test_cycle14_section0_recent_order_new_fill_rules_parity_and_active_plan_rr_invariant() -> None:
    """Verify Cycle 14 fixes:
    1. Section 0 new_fill trigger detects fresh BOT/MANUAL/resting fills from recent_orders (<=20m),
       promotes the filled symbol ahead of BTCUSDC in Section 3, ignores stale fills (>20m),
       deduplicates against last_cycle_execution_status, and formats tiny partial_fill rem without sci notation.
    2. Full rules prose references actual rendered open_limit_orders table labels (Dist, Fill1h, Note, front_run=)
       instead of stale pre-table JSON keys (dist_from_spot_pct, fill_prob_1h_pct, recommended_front_run_px).
    3. audit_rendered_prompt_text validates unlinked Active AI Trade Plans dist% & RR math symmetrically.
    """
    from types import SimpleNamespace
    from botcore import topic_spec
    from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

    t0 = 1_800_000_000.0
    sensors = SimpleNamespace(
        evaluated_at_ts=t0,
        prices_and_trends={
            "BTCUSDC": {"price": 60000.0, "chg_24h_pct": 1.0},
            "ARBUSDC": {"price": 0.1132, "chg_24h_pct": 5.0},
            "TAOUSDC": {"price": 182.8, "chg_24h_pct": 3.0},
        },
        microstructure={},
        portfolio_summary={
            "tradable_positions": [
                {"venue": "binance", "symbol": "BTCUSDC", "price": 60000.0, "free_base": 0.01, "value_usd": 600.0, "free_quote": 1000.0},
                {"venue": "binance", "symbol": "ARBUSDC", "price": 0.1132, "free_base": 2000.0, "value_usd": 226.4, "free_quote": 1000.0},
                {"venue": "binance", "symbol": "TAOUSDC", "price": 182.8, "free_base": 0.0, "value_usd": 0.0, "free_quote": 1000.0},
            ],
            "open_orders": [
                {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "price": 59500.0,
                    "qty": 0.00008,
                    "filled_qty": 0.00001,
                    "status": "PARTIALLY_FILLED",
                    "order_id": "101",
                    "actor": "AI",
                }
            ],
            "last_cycle_execution_status": [
                {"venue": "binance", "symbol": "TAOUSDC", "side": "BUY", "status": "OPEN_RESTING", "limit_px": 180.5, "age_min": 5.0}
            ],
            "recent_orders": {
                "ARBUSDC": {
                    "last_buy": {"price": 0.1125, "qty": 2666.6, "ts_ms": (t0 - 180.0) * 1000.0, "actor": "BOT"},
                    "last_sell": {"price": 0.1093, "qty": 1000.0, "ts_ms": (t0 - 7200.0) * 1000.0, "actor": "AI"},
                },
                "TAOUSDC": {
                    "last_buy": {"price": 171.1, "qty": 4.0, "ts_ms": (t0 - 3600.0) * 1000.0, "actor": "AI"},
                },
            },
        },
        source_ages_sec={},
    )
    txt_cr = build_supervisor_prompt(
        sensors, None, spec=topic_spec("auto_cryptotrade_executor"), compact_rules=False, wake_reason="new_fill"
    )
    assert "ARBUSDC BUY FILLED@0.1125(BOT)" in txt_cr
    assert "0.1093" not in txt_cr.splitlines()[txt_cr.splitlines().index(next(l for l in txt_cr.splitlines() if l.startswith("- Trigger:")))]
    assert "BTCUSDC partial_fill 0.00001/0.00008(rem=0.00007)" in txt_cr
    assert "7e-05" not in txt_cr
    s3_cr = txt_cr[txt_cr.index("### 3. TRADABLE MARKETS"):]
    assert s3_cr.index("  * ARBUSDC:") < s3_cr.index("  * BTCUSDC:")

    # 2. Rules prose matches rendered open_limit_orders table labels in both crypto and equities.
    rules_cr = txt_cr.split("LIVE TELEMETRY SNAPSHOT:", 1)[0]
    for stale_tok in ("dist_from_spot_pct", "fill_prob_1h_pct", "recommended_front_run_px"):
        assert stale_tok not in rules_cr
    assert "(`Dist`, `Fill1h`, `Note`; watch `[ob_stale]` when `ob_age > 30s`)" in rules_cr
    assert "(`Fill1h<=5%` or `|Dist|>4%`) or wall-trapped (`front_run=` / `Note` wall flags)" in rules_cr

    txt_eq = build_supervisor_prompt(
        SimpleNamespace(evaluated_at_ts=t0), None, spec=topic_spec("auto_sharetrade_executor"), compact_rules=False
    )
    rules_eq = txt_eq.split("LIVE TELEMETRY SNAPSHOT:", 1)[0]
    for stale_tok in ("dist_from_spot_pct", "fill_prob_1h_pct", "recommended_front_run_px"):
        assert stale_tok not in rules_eq
    assert "Audit `open_limit_orders` (`Dist`, `Fill1h`)" in rules_eq
    assert "re-anchor stale/far (`Fill1h<=5%` or `|Dist|>4%`)" in rules_eq

    # 3. PROMPT_ACTIVE_PLAN_RR_MATH validates unlinked FILLED & RESTING plans symmetrically (explicit @spot/@limit and legacy bare RR=).
    good_plans = (
        "    - [binance] ARBUSDC BUY@0.1125 (FILLED, spot=0.1132, age=0.4h): target=0.1156 (+2.12% vs_spot), invalidation=0.11 (-2.83% vs_spot), RR@spot=0.75x [ACTIVE]\n"
        "    - [t212] NVDA_US_EQ SELL@155.0 (RESTING, age=1.0h): target=145.0 (-6.45% vs_limit), invalidation=160.0 (+3.23% vs_limit), RR@limit=2.00x [ACTIVE]\n"
    )
    assert audit_rendered_prompt_text(good_plans) == []
    bad_plan_rr = (
        "    - [binance] ARBUSDC BUY@0.1125 (FILLED, spot=0.1132, age=0.4h): target=0.1156 (+5.00% vs_spot), invalidation=0.11 (-2.83% vs_spot), RR@spot=1.99x [ACTIVE]\n"
    )
    bad_vs = [v for v in audit_rendered_prompt_text(bad_plan_rr) if v.code == "PROMPT_ACTIVE_PLAN_RR_MATH"]
    assert len(bad_vs) == 2
    bad_anchor_tag = (
        "    - [binance] ARBUSDC BUY@0.1125 (FILLED, spot=0.1132, age=0.4h): target=0.1156 (+2.12% vs_spot), invalidation=0.11 (-2.83% vs_spot), RR@limit=0.75x [ACTIVE]\n"
    )
    bad_anc_vs = [v for v in audit_rendered_prompt_text(bad_anchor_tag) if v.code == "PROMPT_ACTIVE_PLAN_RR_MATH"]
    assert len(bad_anc_vs) == 1 and "rr_anchor" in bad_anc_vs[0].subject

    # 4. Linked open-order plan aligns dist% and RR@limit= to displayed exchange order price o_px even after an amend drift.
    from orchestratorOS.supervisor.prompt_sections import _format_open_limit_orders_table

    oo_amended_lines = "\n".join(
        _format_open_limit_orders_table(
            [
                {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "price": 60000.0,
                    "qty": 0.01,
                    "filled_qty": 0.0,
                    "order_id": "202",
                    "actor": "AI",
                    "age": "0.5h",
                    "dist_from_spot_pct": -0.5,
                    "fill_prob_1h_pct": 40,
                }
            ],
            active_plans=[
                {
                    "venue": "binance",
                    "symbol": "BTCUSDC",
                    "side": "BUY",
                    "order_state": "RESTING",
                    "entry_px": 59500.0,
                    "target_px": 61200.0,
                    "target_dist_pct": 2.86,
                    "invalidation_px": 58800.0,
                    "invalidation_dist_pct": -1.18,
                    "status": "ACTIVE",
                }
            ],
        )
    )
    assert "61200.0(+2.00%)/58800.0(-2.00%) RR@limit=1.00x [ACTIVE]" in oo_amended_lines
    assert audit_rendered_prompt_text(oo_amended_lines) == []
    bad_oo_pct = oo_amended_lines.replace("61200.0(+2.00%)", "61200.0(+5.00%)")
    assert any(v.code == "PROMPT_OPEN_ORDER_RR_MATH" and v.subject.endswith(":target") for v in audit_rendered_prompt_text(bad_oo_pct))


def test_full_mathematical_audit_snapshot_and_rendered_prompt_invariants() -> None:
    """Verify full mathematical audit suite across snapshot and rendered prompt text (clean passes, every corruption caught)."""
    clean_snap = {
        "prices_and_trends": {
            "BTCUSDC": {
                "price": 60000.0,
                "chg_15m_pct": 0.20,
                "chg_1h_pct": 1.00,
                "chg_24h_pct": 4.00,
                "low_15m": 59800.0,
                "high_15m": 60200.0,
                "pos_15m_pct": 50.0,
                "low_24h": 57000.0,
                "high_24h": 61000.0,
                "pos_24h_pct": 75.0,
                "dd_24h_high_pct": -1.64,
                "bounce_24h_low_pct": 5.26,
                "span_24h_pct": 7.02,
                "swing_low_px": 57000.0,
                "swing_high_px": 61000.0,
                "dist_vs_session_vwap_pct": 0.50,
                "session_vwap_z": 1.20,
            },
            "ARBUSDC": {
                "price": 0.1132,
                "chg_15m_pct": 0.50,
                "chg_1h_pct": 0.40,
                "chg_24h_pct": 6.50,
                "vs_btc_15m_pct": 0.30,
                "vs_btc_1h_pct": -0.60,
                "vs_btc_24h_pct": 2.50,
                "low_15m": 0.1125,
                "high_15m": 0.1135,
                "pos_15m_pct": 70.0,
                "low_24h": 0.1050,
                "high_24h": 0.1160,
                "pos_24h_pct": 74.55,
                "dd_24h_high_pct": -2.41,
                "bounce_24h_low_pct": 7.81,
                "span_24h_pct": 10.48,
                "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
                "prior_6h_low_px": 0.1110,
                "recent_1h_low_px": 0.1120,
                "dist_vs_session_vwap_pct": -0.80,
                "session_vwap_z": -1.40,
            },
            "SPCXxUSD": {"price": 129.85},
        },
        "microstructure": {
            "BTCUSDC": {
                "best_bid": 59999.0,
                "best_ask": 60001.0,
                "ob_mid_price": 60000.0,
                "spread_bps": 0.33,
                "bid_wall_px": 59700.0,
                "bid_wall_dist_pct": -0.50,
                "ask_wall_px": 60300.0,
                "ask_wall_dist_pct": 0.50,
            },
        },
        "portfolio_summary": {
            "tradable_positions": [
                {
                    "venue": "binance",
                    "symbol": "ARBUSDC",
                    "price": 0.1132,
                    "free_base": 2000.0,
                    "value_usd": 226.40,
                    "active_avg_buy_px": 0.1125,
                    "pnl_vs_avg_pct": 0.62,
                    "trailing_peak": 0.1150,
                    "dd_from_peak_pct": -1.57,
                },
                {
                    "venue": "kraken",
                    "symbol": "SPCXxUSD",
                    "price": 129.85,
                    "free_base": 10.0,
                    "value_usd": 1298.50,
                    "active_avg_buy_px": 134.84,
                    "pnl_vs_avg_pct": -3.70,
                },
            ],
            "kraken_cycles": {"SPCXxUSD": {"avg_cost_px": 134.84}},
            "open_limit_orders": [
                {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 59400.0, "dist_from_spot_pct": -1.00}
            ],
            "kraken_xstocks": {"listed_pair": "SPCXxUSD", "underlying_px": 128.95, "xstock_vs_underlying_pct": 0.70},
        },
    }
    assert audit_snapshot_invariants(clean_snap) == []

    # Corrupt every mathematical relation in snapshot and assert every invariant code fires.
    bad_snap = {
        "prices_and_trends": {
            "BTCUSDC": {
                "price": 60000.0,
                "chg_1h_pct": 1.00,
                "low_24h": 57000.0,
                "high_24h": 61000.0,
                "dd_24h_high_pct": -5.00,  # wrong (exp -1.64%)
                "bounce_24h_low_pct": 1.00,  # wrong (exp +5.26%)
                "span_24h_pct": 2.00,  # wrong (exp 7.02%)
                "swing_low_px": 60500.0,  # price below swing_low
                "swing_high_px": 61000.0,
                "dist_vs_session_vwap_pct": 0.80,
                "session_vwap_z": -1.50,  # opposite sign
            },
            "ARBUSDC": {
                "price": 0.1132,
                "chg_1h_pct": 0.40,
                "vs_btc_1h_pct": 2.50,  # wrong (exp -0.60%)
                "low_15m": 0.1120,
                "low_24h": 0.1100,
                "bottom_structure_6h": "HIGHER_LOW_DOUBLE_BOTTOM",
                "prior_6h_low_px": 0.1130,
                "recent_1h_low_px": 0.1125,  # > low_15m AND < prior_6h_low
            },
            "SPCXxUSD": {"price": 129.85},
        },
        "microstructure": {
            "BTCUSDC": {
                "best_bid": 60010.0,
                "best_ask": 60000.0,  # crossed book
                "ob_mid_price": 60000.0,
                "spread_bps": 25.0,  # wrong spread
                "bid_wall_px": 59700.0,
                "bid_wall_dist_pct": -4.00,  # wrong wall dist (exp -0.50%)
            },
        },
        "portfolio_summary": {
            "tradable_positions": [
                {
                    "venue": "binance",
                    "symbol": "ARBUSDC",
                    "price": 0.1132,
                    "free_base": 2000.0,
                    "value_usd": 300.00,  # wrong (exp 226.40)
                    "active_avg_buy_px": 0.1125,
                    "pnl_vs_avg_pct": 9.99,  # wrong
                    "trailing_peak": 0.1100,  # px > peak
                    "dd_from_peak_pct": -5.00,  # wrong
                },
                {
                    "venue": "kraken",
                    "symbol": "SPCXxUSD",
                    "price": 129.85,
                    "free_base": 10.0,
                    "value_usd": 1298.50,
                    "active_avg_buy_px": 134.84,
                },
            ],
            "kraken_cycles": {"SPCXxUSD": {"avg_cost_px": 140.00}},  # diverges from position avg_cost
            "open_limit_orders": [
                {"venue": "binance", "symbol": "BTCUSDC", "side": "BUY", "price": 59400.0, "dist_from_spot_pct": -5.00}
            ],
            "kraken_xstocks": {"listed_pair": "SPCXxUSD", "underlying_px": 128.95, "xstock_vs_underlying_pct": 5.50},
        },
    }
    bad_snap_codes = {v.code for v in audit_snapshot_invariants(bad_snap)}
    assert {
        "POSITION_VALUE_AND_ANCHOR_MATH",
        "OPEN_ORDER_DIST_MATH",
        "XSTOCK_PREMIUM_MATH",
        "VS_BTC_MATH",
        "RANGE_BOUNDS_AND_EXTREMES_MATH",
        "BOTTOM_STRUCTURE_MATH",
        "SESSION_VWAP_SIGN_MATH",
        "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
    }.issubset(bad_snap_codes)

    # Rendered prompt text mathematical audit: clean prompt passes with 0 violations.
    clean_prompt = (
        "LIVE TELEMETRY SNAPSHOT:\n"
        "### 1. PORTFOLIO:\n"
        "    - [binance] ARBUSDC: free_ARB=2000.0 ($226.40 USD), px=0.1132, avg_cost=0.1125 (pnl_vs_cost=0.62%), peak_px=0.115 (dd_from_peak=-1.57%), limits(max_buy=$1000.0, max_sell=$226.4)\n"
        "    - [kraken] SPCXxUSD: free_SPCXx=10.0 ($1298.50 USD), px=129.85, avg_cost=134.84 (pnl_vs_cost=-3.70%), limits(max_buy=$0.0, max_sell=$1298.5)\n"
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |\n"
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "    | binance | BTCUSDC | BUY | 59400.0 | 0.01 | 0 | 1 | AI | 1.0h | -1.00% | 20% | 60600.0(+2.02%)/58800.0(-1.01%) RR@limit=2.00x [ACTIVE] | - |\n"
        "    - [kraken] SPCXxUSD: qty=10.0, cost=$1348.40, avg_cost=134.84, pnl_vs_cost=-3.70%\n"
        "  * Kraken xStocks: SPCX=$128.95, EXT_CLOSE, xstock_vs_underlying=+0.70%\n"
        "### 3. TRADABLE MARKETS:\n"
        "  * BTCUSDC: Price=$60,000.00\n"
        "    - Trend: 15m=+0.20% 1h=+1.00% 24h=+4.00% 24h_range=[57000.0, 61000.0] pos_24h=75.0% (dd_high=-1.64%, bounce_low=+5.26%, span=7.02%) | session_vwap=$59,700.00 (+0.50%, z=+1.20, utc_00) | atr(15m=0.28%, 1h=0.88%, 4h=1.74%)\n"
        "    - Book: ob_imbalance=0.500 (bid_depth=$100.0K, ask_depth=$100.0K), bid_wall=59700.0 ($50.0K, dist=-0.50%), ask_wall=60300.0 ($50.0K, dist=+0.50%), spread=0.33bps (bid=59999.0/ask=60001.0), ob_mid=60000.0\n"
        "  * ARBUSDC: Price=$0.1132\n"
        "    - Trend: 15m=+0.50% 1h=+0.40% 24h=+6.50% 24h_range=[0.105, 0.116] pos_24h=74.5% (dd_high=-2.41%, bounce_low=+7.81%, span=10.48%) | 15m_range=[0.1125, 0.1135] pos_15m=70.0% | vs_btc(15m=+0.30%, 1h=-0.60%, 24h=+2.50%) | bottom_6h(prior_low=0.111, 1h_low=0.112, higher_low_pct=+0.90%)\n"
        "  * SPCXxUSD: Price=$129.85\n"
        "    - Trend: 24h=+1.00%\n"
    )
    assert audit_rendered_prompt_text(clean_prompt) == []

    # Corrupt every rendered prompt mathematical invariant and verify all codes fire.
    bad_prompt = (
        "LIVE TELEMETRY SNAPSHOT:\n"
        "### 1. PORTFOLIO:\n"
        "    - [binance] ARBUSDC: free_ARB=2000.0 ($350.00 USD), px=0.1132, avg_cost=0.1125 (pnl_vs_cost=9.50%), peak_px=0.110 (dd_from_peak=-5.00%), limits(max_buy=$1000.0, max_sell=$226.4)\n"
        "    - [kraken] SPCXxUSD: free_SPCXx=10.0 ($1298.50 USD), px=129.85, avg_cost=134.84 (pnl_vs_cost=-3.70%), limits(max_buy=$0.0, max_sell=$1298.5)\n"
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |\n"
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "    | binance | BTCUSDC | BUY | 59400.0 | 0.01 | 0 | 1 | AI | 1.0h | -5.50% | 20% | 60600.0(+2.02%)/58800.0(-1.01%) RR@limit=2.00x [ACTIVE] | - |\n"
        "    - [kraken] SPCXxUSD: qty=10.0, cost=$1348.40, avg_cost=141.00, pnl_vs_cost=-3.70%\n"
        "  * Kraken xStocks: SPCX=$128.95, EXT_CLOSE, xstock_vs_underlying=+4.20%\n"
        "### 3. TRADABLE MARKETS:\n"
        "  * BTCUSDC: Price=$60,000.00\n"
        "    - Trend: 15m=+0.20% 1h=+1.00% 24h=+4.00% 24h_range=[57000.0, 61000.0] pos_24h=75.0% (dd_high=-4.00%, bounce_low=+1.00%, span=2.00%) | session_vwap=$59,700.00 (+0.50%, z=-1.20, utc_00)\n"
        "    - Book: ob_imbalance=0.500 (bid_depth=$100.0K, ask_depth=$100.0K), bid_wall=59700.0 ($50.0K, dist=-3.50%), spread=15.0bps (bid=59999.0/ask=60001.0), ob_mid=60000.0\n"
        "  * ARBUSDC: Price=$0.1132\n"
        "    - Trend: 15m=+0.50% 1h=+0.40% 24h=+6.50% 24h_range=[0.105, 0.116] pos_24h=74.5% (dd_high=-2.41%, bounce_low=+7.81%, span=10.48%) | 15m_range=[0.1120, 0.1135] pos_15m=80.0% | vs_btc(1h=+3.00%) | bottom_6h(prior_low=0.111, 1h_low=0.1125, higher_low_pct=+5.00%)\n"
        "  * SPCXxUSD: Price=$129.85\n"
        "    - Trend: 24h=-0.00%, bad_val=None\n"
    )
    bad_prompt_codes = {v.code for v in audit_rendered_prompt_text(bad_prompt)}
    assert {
        "PROMPT_POSITION_MATH",
        "PROMPT_CYCLE_ANCHOR_PARITY",
        "PROMPT_OPEN_ORDER_DIST_MATH",
        "PROMPT_XSTOCK_PREMIUM_MATH",
        "PROMPT_RANGE_EXTREMES_MATH",
        "PROMPT_SESSION_VWAP_DELTA",
        "PROMPT_OB_SPREAD_AND_WALL_DIST",
        "PROMPT_VS_BTC_MATH",
        "PROMPT_BOTTOM_STRUCTURE_MATH",
        "PROMPT_NUMERIC_HYGIENE",
    }.issubset(bad_prompt_codes)

    # MicrostructurePromptViewEvaluator must suppress whale_long_pct when absent so prompts never emit '=None'.
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_microstructure_prompt_view

    rows_no_whale = evaluate_microstructure_prompt_view(
        {"BTCUSDC": {"ob_imbalance": 0.5, "bid_depth_usd": 100000.0, "ask_depth_usd": 100000.0, "ob_mid_price": 60000.0}},
        prices_and_trends={"BTCUSDC": {"price": 60000.0}},
    )
    assert rows_no_whale and rows_no_whale[0]["show_whale_long"] is False


def test_cross_section_prompt_invariants_and_live_evaluator() -> None:
    """Verify all 6 cross-section mathematical invariants (clean passes, every corruption caught) + live deduplication."""
    clean_prompt = (
        "LIVE TELEMETRY SNAPSHOT:\n"
        "### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:\n"
        "  * Portfolio Risk Engine (Crypto Scope): crypto_nav=$10,000.00, invested=$4,000.00, "
        "free_cash=$5,000.00 (50.0%), locked_quote=$1,000.00 (10.0%), venues(binance=$7,000.00, kraken=$3,000.00)\n"
        "  * Available Free Quote Cash per Venue: binance USDC=$3,000.00 [locked_quote_usd=$1,000.00]; kraken USD=$2,000.00\n"
        "  * Tradable Spot Positions:\n"
        "    - [binance] BTCUSDC: free_BTC=0.05 ($3,000.00 USD), px=60000.0, limits(max_buy=$2000.0, max_sell=$3000.0)\n"
        "    - [kraken] HYPEUSD: free_HYPE=20.0 ($1,000.00 USD), px=50.0, limits(max_buy=$2000.0, max_sell=$1000.0)\n"
        "  * Open Limit Orders:\n"
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |\n"
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "    | binance | BTCUSDC | BUY | 59400.0 | 0.015 | 0 | 11 | AI | 1.0h | -1.00% | 20% | 60600.0(+2.02%)/58800.0(-1.01%) RR@limit=2.00x [ACTIVE] | DEEP_PASSIVE_BID(wall=59700.0/$50.0K), front_run=59750.0 |\n"
        "  * Venue Specs:\n"
        "    | Venue | Symbol | Tier | Fee bps (M/T) | MinNotional | MinQty | Step | Tick |\n"
        "    |---|---|---|---|---|---|---|---|\n"
        "    | hyperliquid | HYPEUSDC | TIER_0 | 4 (4.0/4.5) | $10 | 0.01 | 0.01 | 0.01 |\n"
        "    | kraken | HYPEUSD | TIER_0 | 22 (22.0/37.0) | $1 | 0.06 | 0.00000001 | 0.01 |\n"
        "  * AI Execution & Plan Calibration (7d): plans=10 (BUY:3hit/1inv/1early/1open, SELL:2hit/1inv/0early/1open, hit=71%) | "
        "binance: LIM 4/5[80%] | fill_by_offset(<0.2%:100%[2/2], 0.2-0.5%:50%[2/4])\n"
        "### 2. MACRO, REGIME & CROSS-MARKET:\n"
        "  * Market Breadth: advance_ratio=0.4138, advancing=12/29, BTC_vol_share=38.5%, Alt_vol_share=61.5%\n"
        "  * Macro Sentiment Advisor (conf=0.75): Neutral | live_btc_flow_delta=(BTC=$60,000.00(POC=$59,500.00), taker_ratio=1.20)\n"
        "### 3. TRADABLE MARKETS:\n"
        "  * BTCUSDC: Price=$60,000.00\n"
        "    - Trend: 24h=+1.00% | vol_profile(POC=59500.0[+0.84%], VAL=59000.0[+1.69%], VAH=60500.0[-0.83%], va_pos=66.7%)\n"
        "    - Book: ob_imbalance=0.500 (bid_depth=$100.0K, ask_depth=$100.0K), depth_1pct=(bid=$60.0K/ask=$50.0K), "
        "bid_wall=59700.0 ($50.0K, dist=-0.50%), ask_wall=60300.0 ($50.0K, dist=+0.50%), spread=0.33bps (bid=59999.0/ask=60001.0), ob_mid=60000.0\n"
        "    - Flow: taker_ratio=1.20 (buy_v=$1.20M/sell_v=$1.00M), cvd_15m=+1.00 BTC (+$60.0K)\n"
        "  * HYPEUSDC: Price=$50.01\n"
        "    - Trend: 24h=+2.00%\n"
        "    - Flow: arb_vs_HYPEUSD=+2.0bps, net_buy_edge=+16.0bps[maker]\n"
        "  * HYPEUSD: Price=$50.00\n"
        "    - Trend: 24h=+2.00%\n"
        "    - Flow: arb_vs_HYPEUSDC=-2.0bps, net_buy_edge=-16.0bps[maker]\n"
    )
    assert audit_rendered_prompt_text(clean_prompt) == []

    bad_prompt = (
        "LIVE TELEMETRY SNAPSHOT:\n"
        "### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:\n"
        "  * Portfolio Risk Engine (Crypto Scope): crypto_nav=$10,000.00, invested=$4,000.00, "
        "free_cash=$5,000.00 (50.0%), locked_quote=$1,000.00 (10.0%), venues(binance=$7,000.00, kraken=$3,000.00)\n"
        "  * Available Free Quote Cash per Venue: binance USDC=$4,200.00 [locked_quote_usd=$1,500.00]; kraken USD=$2,000.00\n"
        "  * Tradable Spot Positions:\n"
        "    - [binance] BTCUSDC: free_BTC=0.05 ($3,000.00 USD), px=60000.0, limits(max_buy=$2000.0, max_sell=$3000.0)\n"
        "    - [kraken] HYPEUSD: free_HYPE=40.0 ($2,000.00 USD), px=50.0, limits(max_buy=$2000.0, max_sell=$2000.0)\n"
        "  * Open Limit Orders:\n"
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |\n"
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "    | binance | BTCUSDC | BUY | 59400.0 | 0.03 | 0 | 11 | AI | 1.0h | -1.00% | 20% | 60600.0(+2.02%)/58800.0(-1.01%) RR@limit=2.00x [ACTIVE] | DEEP_PASSIVE_BID(wall=58000.0/$50.0K), front_run=60500.0 |\n"
        "  * Venue Specs:\n"
        "    | Venue | Symbol | Tier | Fee bps (M/T) | MinNotional | MinQty | Step | Tick |\n"
        "    |---|---|---|---|---|---|---|---|\n"
        "    | hyperliquid | HYPEUSDC | TIER_0 | 4 (4.0/4.5) | $10 | 0.01 | 0.01 | 0.01 |\n"
        "    | kraken | HYPEUSD | TIER_0 | 22 (22.0/37.0) | $1 | 0.06 | 0.00000001 | 0.01 |\n"
        "  * AI Execution & Plan Calibration (7d): plans=12 (BUY:3hit/1inv/1early/1open, SELL:2hit/1inv/0early/1open, hit=95%) | "
        "binance: LIM 4/5[20%] | fill_by_offset(<0.2%:10%[2/2], 0.2-0.5%:50%[2/4])\n"
        "### 2. MACRO, REGIME & CROSS-MARKET:\n"
        "  * Market Breadth: advance_ratio=0.7500, advancing=12/29, BTC_vol_share=65.0%, Alt_vol_share=55.0%\n"
        "  * Macro Sentiment Advisor (conf=0.75): Neutral | live_btc_flow_delta=(BTC=$62,500.00(POC=$58,000.00), taker_ratio=0.75)\n"
        "### 3. TRADABLE MARKETS:\n"
        "  * BTCUSDC: Price=$60,000.00\n"
        "    - Trend: 24h=+1.00% | vol_profile(POC=59500.0[+0.84%], VAL=59000.0[+1.69%], VAH=60500.0[-0.83%], va_pos=66.7%)\n"
        "    - Book: ob_imbalance=0.500 (bid_depth=$100.0K, ask_depth=$100.0K), depth_1pct=(bid=$150.0K/ask=$150.0K), "
        "bid_wall=59700.0 ($50.0K, dist=-0.50%), ask_wall=60300.0 ($50.0K, dist=+0.50%), spread=0.33bps (bid=59999.0/ask=60001.0), ob_mid=60000.0\n"
        "    - Flow: taker_ratio=2.50 (buy_v=$1.20M/sell_v=$1.00M), cvd_15m=+1.00 BTC (-$500.0K)\n"
        "  * HYPEUSDC: Price=$50.01\n"
        "    - Trend: 24h=+2.00%\n"
        "    - Flow: arb_vs_HYPEUSD=+2.0bps, net_buy_edge=+5.0bps[maker]\n"
        "  * HYPEUSD: Price=$50.00\n"
        "    - Trend: 24h=+2.00%\n"
        "    - Flow: arb_vs_HYPEUSDC=-2.0bps, net_buy_edge=+5.0bps[maker]\n"
    )
    bad_vs = audit_rendered_prompt_text(bad_prompt)
    bad_codes = {v.code for v in bad_vs}
    assert {
        "PROMPT_SECTION1_SUM_CONSERVATION",
        "PROMPT_OPEN_ORDER_WALL_BOOK_MATH",
        "PROMPT_ADVISOR_BTC_PARITY",
        "PROMPT_FLOW_DEPTH_EDGE_MATH",
        "PROMPT_CALIBRATION_MATH",
        "PROMPT_BREADTH_MATH",
    }.issubset(bad_codes)
    # All 4 Section 1 sub-checks fired independently
    s1_subjects = {v.subject for v in bad_vs if v.code == "PROMPT_SECTION1_SUM_CONSERVATION"}
    assert {
        "Section1.InvestedSum",
        "Section1.FreeCashSum",
        "Section1.LockedQuoteSum",
        "Section1.OpenBuyVsLockedQuote",
    }.issubset(s1_subjects)

    # evaluate_live_prompt_invariants deduplicates identical violations across full + warm prompts
    warm_extra = bad_prompt + "\n    - Trend: 24h=+1.00%, bad_warm=None\n"
    combined = evaluate_live_prompt_invariants({}, bad_prompt, resume_prompt_text=warm_extra)
    assert len(combined) == len(bad_vs) + 1
    assert any(v.code == "PROMPT_NUMERIC_HYGIENE" for v in combined)


def test_reconciler_continuous_runtime_prompt_invariant_validation(tmp_path, caplog) -> None:
    """Verify AutonomousAIReconciler validates rendered prompts inline before LLM calls, logs ERROR, persists on report, and rate-limits ntfy across runs."""
    import logging
    from unittest.mock import patch

    from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
    from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
    from orchestratorOS.supervisor.autonomous_ai_reconciler import AutonomousAIReconciler

    ws = tmp_path / "ws_inv"
    (ws / "cachedb").mkdir(parents=True)
    (ws / "order_guard.conf").write_text(
        "supervisor_mode = shadow\n"
        "supervisor_notify_enabled = true\n"
        "ntfy_url = https://ntfy.example.test\n"
        "ntfy_topic_server = srv-topic\n"
        "ntfy_topic_error = err-topic\n"
        "ntfy_topic_guard = grd-topic\n"
        "autonomous_trading_min_notional_usd = 10.0\n"
        "autonomous_trading_max_notional_usd = 2000.0\n"
        "autonomous_trading_min_confidence = 0.85\n",
        encoding="utf-8",
    )

    def _make_snap(*, bad_breadth: bool = False) -> TelemetrySensorSnapshot:
        snap = TelemetrySensorSnapshot()
        snap.evaluated_at_ts = 1_800_000_000.0
        snap.market_breadth = {
            "advance_ratio": 0.9000 if bad_breadth else 0.4138,
            "advancing": 12,
            "total_symbols": 29,
        }
        snap.prices_and_trends = {"BTCUSDC": {"price": 60000.0, "chg_24h_pct": 1.0}}
        snap.portfolio_summary = {
            "portfolio_risk_engine": {
                "unified_nav_usd": 5000.0,
                "invested_exposure_usd": 3000.0,
                "global_free_cash_usd": 2000.0,
                "global_free_cash_pct": 40.0,
                "venue_nav_breakdown_usd": {"binance": 5000.0},
            },
            "venue_quote_cash": {"binance": {"venue": "binance", "asset": "USDC", "free": 2000.0}},
            "tradable_positions": [
                {"venue": "binance", "symbol": "BTCUSDC", "free_base": 0.05, "value_usd": 3000.0, "free_quote": 2000.0, "price": 60000.0}
            ],
        }
        return snap

    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)
    ntfy_calls: list[tuple[str, str]] = []

    def _fake_ntfy(title, body, *_args, **_kwargs):
        ntfy_calls.append((title, body))
        return True

    fake_resp = {"summary": "Hold steady.", "market_trading_decisions": [{"decision": "HOLD", "confidence": 0.0}]}
    rec = AutonomousAIReconciler(mode="shadow", workspace_dir=str(ws), notify_enabled=True)

    with (
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.query_topic_json", return_value=fake_resp),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.send_supervisor_ntfy_http", side_effect=_fake_ntfy),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.get_topic_thread_state", return_value=("cid-1", 0, False)),
        caplog.at_level(logging.ERROR, logger="orchestratorOS.supervisor"),
    ):
        # 1. Clean snapshot + prompt -> zero violations, zero logs, zero ntfy alerts
        rec.synthesize_and_diagnose(_make_snap(bad_breadth=False), logs, topic_key="auto_cryptotrade_executor")
        assert rec._last_prompt_invariant_violations == []
        assert ntfy_calls == []
        assert "[PROMPT_INVARIANT_VIOLATION]" not in caplog.text

        # 2. Corrupted breadth -> detects violation, logs ERROR for auto_technical_repair, sends 1 ntfy alert
        rec.synthesize_and_diagnose(_make_snap(bad_breadth=True), logs, topic_key="auto_cryptotrade_executor")
        assert any("PROMPT_BREADTH_MATH" in line for line in rec._last_prompt_invariant_violations)
        assert "[PROMPT_INVARIANT_VIOLATION]" in caplog.text
        assert len(ntfy_calls) == 1
        assert "PROMPT_BREADTH_MATH" in ntfy_calls[0][1]

        # 3. Immediate repeat within 4h cooldown -> still records violation, suppresses duplicate ntfy alert
        rec.synthesize_and_diagnose(_make_snap(bad_breadth=True), logs, topic_key="auto_cryptotrade_executor")
        assert any("PROMPT_BREADTH_MATH" in line for line in rec._last_prompt_invariant_violations)
        assert len(ntfy_calls) == 1

    # 4. Cross-process persistence via run_cycle(): SupervisorAuditReport persists violations + alert cooldowns
    rec2 = AutonomousAIReconciler(mode="shadow", workspace_dir=str(ws), notify_enabled=True)
    with (
        patch.object(rec2, "harvest_sensors_telemetry", return_value=_make_snap(bad_breadth=True)),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.query_topic_json", return_value=fake_resp),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.send_supervisor_ntfy_http", side_effect=_fake_ntfy),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.get_topic_thread_state", return_value=("cid-1", 0, False)),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.bool_env", return_value=False),
    ):
        rep1 = rec2.run_cycle(topic_key="auto_cryptotrade_executor")
        assert any("PROMPT_BREADTH_MATH" in line for line in rep1.prompt_invariant_violations)
        assert rep1.prompt_invariant_alert_ts
        assert len(ntfy_calls) == 2

    # Fresh process reading persisted {topic_key}_eval.json honors cross-process 4h cooldown
    rec3 = AutonomousAIReconciler(mode="shadow", workspace_dir=str(ws), notify_enabled=True)
    with (
        patch.object(rec3, "harvest_sensors_telemetry", return_value=_make_snap(bad_breadth=True)),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.query_topic_json", return_value=fake_resp),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.send_supervisor_ntfy_http", side_effect=_fake_ntfy),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.get_topic_thread_state", return_value=("cid-1", 0, False)),
        patch("orchestratorOS.supervisor.autonomous_ai_reconciler.bool_env", return_value=False),
    ):
        rep2 = rec3.run_cycle(topic_key="auto_cryptotrade_executor")
        assert any("PROMPT_BREADTH_MATH" in line for line in rep2.prompt_invariant_violations)
        assert len(ntfy_calls) == 2


def test_cycle15_fill_lifecycle_pruning_lot_harvested_and_four_new_invariants() -> None:
    """Verify Cycle 15 fixes and 4 new continuous invariants:
    1. prune_reconciled_terminal_open_orders drops stale open-order ghosts once reconciled terminal in intents,
       and format_prompt_intent_status preserves FILLED over a lagging in_open_book=True flag.
    2. is_last_buy_lot_harvested & PortfolioPromptViewEvaluator never label a fresh BUY newer than the last SELL
       as lot_harvested even while balance cache still shows dust.
    3. Equities Cross-Venue Kraken xStock reference rounds alloc_price cleanly to <=6 decimals.
    4. PROMPT_VA_POS_AND_LIQ_MATH, PROMPT_ORDER_LIFECYCLE_CONSISTENCY, PROMPT_WARM_ELISION_TRUTHFULNESS,
       and PROMPT_LONG_HORIZON_MATH pass cleanly on valid prompts and catch every corruption.
    """
    from intelligence.sentiment.evaluators.portfolio_evaluator import is_last_buy_lot_harvested
    from intelligence.sentiment.evaluators.prompt_view_evaluator import evaluate_portfolio_prompt_view
    from orchestratorOS.supervisor.prompt_sections import _format_kraken_xstocks_line
    from providers.intent_state import prune_reconciled_terminal_open_orders
    from providers.order_status_ssot import format_prompt_intent_status

    # 1a. format_prompt_intent_status monotonic lattice: terminal FILLED is never downgraded by stale in_open_book=True
    assert (
        format_prompt_intent_status(
            "EXECUTED", "FILLED", in_open_book=True, filled_qty=16.8, section="last_cycle_execution_status"
        )
        == "FILLED"
    )
    assert (
        format_prompt_intent_status(
            "EXECUTED", "NEW", in_open_book=True, filled_qty=5.0, section="last_cycle_execution_status"
        )
        == "PARTIALLY_FILLED"
    )

    # 1b. prune_reconciled_terminal_open_orders drops stale HYPEUSDC ghost while keeping live BTCUSDC order
    open_orders = [
        {"venue": "binance", "symbol": "BTCUSDC", "order_id": "10976647800", "status": "NEW"},
        {"venue": "hyperliquid", "symbol": "HYPEUSDC", "order_id": "617202101915", "status": "OPEN"},
    ]
    intents = [
        {
            "venue": "hyperliquid",
            "symbol": "HYPEUSDC",
            "status": "EXECUTED",
            "execution_payload": {
                "orderId": "617202101915",
                "status": "FILLED",
                "executedQty": "16.8",
                "reconciled_terminal": True,
            },
        },
        {
            "venue": "binance",
            "symbol": "BTCUSDC",
            "status": "EXECUTED",
            "execution_payload": {"orderId": "10976647800", "status": "NEW", "executedQty": "0"},
        },
    ]
    pruned = prune_reconciled_terminal_open_orders(open_orders, intents)
    assert [o["order_id"] for o in pruned] == ["10976647800"]

    # 2. Fresh BUY fill newer than last SELL is never labeled lot_harvested even when balance cache still shows dust
    now_ms = 1_800_000_000_000
    assert (
        is_last_buy_lot_harvested(
            59.5, [], last_buy_ts_ms=now_ms, last_sell_ts_ms=now_ms - 86_400_000, is_flat_dust=True
        )
        is False
    )
    assert (
        is_last_buy_lot_harvested(
            59.5, [], last_buy_ts_ms=now_ms - 86_400_000, last_sell_ts_ms=now_ms, is_flat_dust=True
        )
        is True
    )
    pv = evaluate_portfolio_prompt_view(
        {
            "tradable_positions": [
                {
                    "venue": "hyperliquid",
                    "symbol": "HYPEUSDC",
                    "price": 59.48,
                    "free_base": 0.0068,
                    "value_usd": 0.41,
                    "free_quote": 3100.0,
                    "last_buy_px": 59.5,
                    "last_buy_ts_ms": now_ms,
                    "last_sell_px": 57.785,
                    "last_sell_ts_ms": now_ms - 80_000_000,
                }
            ]
        },
        asset_scope="crypto",
    )
    lb_anchor = next(a for a in pv["tradable_positions"][0]["anchors"] if a["lbl"] == "last_buy_px")
    assert lb_anchor["lot_harvested"] is False

    # 3. Equities xStocks sibling reference rounds alloc_price cleanly to 6 decimals
    pv_xs = evaluate_portfolio_prompt_view(
        {
            "kraken_xstocks": {
                "listed_pair": "SPCXxUSD",
                "live_price": 129.05,
                "alloc_price": 134.84400571,
                "pnl_vs_alloc_pct": -4.30,
            }
        },
        asset_scope="equities",
    )
    xs_line = _format_kraken_xstocks_line(pv_xs["kraken_xstocks"], scope_s="equities")
    assert "vs alloc 134.844006)" in xs_line
    assert "134.84400571" not in xs_line

    # 4. All 4 new continuous invariants pass on clean prompt and catch every corruption
    clean_c15 = (
        "LIVE TELEMETRY SNAPSHOT:\n"
        "### 0. CLOCK & SNAPSHOT: 2026-11-26 17:15:00 EET (Thu)\n"
        "- Unchanged in thread (active from prior turn): Calendar(1/3), Cycle_Anchors(1/3)\n"
        "### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:\n"
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |\n"
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "    | binance | BTCUSDC | BUY | 59680.0 | 0.015 | 0 | 10976647800 | AI | 1.0h | -0.53% | 80% | 60850.0(+1.96%)/58500.0(-1.98%) RR@limit=0.99x [ACTIVE] | - |\n"
        "  * Last Cycle Execution Status: [hyperliquid] HYPEUSDC BUY LIMIT FILLED (fill_px=59.5, qty=16.8, id=617202101915, age=0.2m)\n"
        "  * Recent Autonomous Intents:\n"
        "    - [hyperliquid] HYPEUSDC BUY LIMIT FILLED (fill_px=59.5, qty=16.8, id=617202101915, age=0.0h)\n"
        "### 2. MACRO, REGIME & CROSS-MARKET:\n"
        "  * Multi-Month & Annual Structure (3y / 36m lookback):\n"
        "    | Symbol | YTD | 30d | 90d | 52w Range (Pos52w) | 3y Range (Pos) | vs MA50 | vs MA200 | Swings (3y, YY-MM:px) |\n"
        "    |---|---|---|---|---|---|---|---|---|\n"
        "    | BTCUSDC | -31.5% | -25.8% | -23.0% | [$55.0K..$95.0K] (12.5%) | [$40.0K..$120.0K] (25.0%) | -19.8% | -16.8% | 23-11:40.0K->25-10:120.0K->now:60.0K |\n"
        "### 3. TRADABLE MARKETS:\n"
        "  * BTCUSDC: Price=$60,000.00\n"
        "    - Trend: 24h=+1.00% | vol_profile(POC=59500.0[+0.84%], VAL=59000.0[+1.69%], VAH=61000.0[-1.64%], va_pos=50.0%, vp_source=REAL_VOL)\n"
        "    - Book: ob_imbalance=0.500 (bid_depth=$100.0K, ask_depth=$100.0K), spread=0.33bps (bid=59999.0/ask=60001.0), ob_mid=60000.0, "
        "liq_clusters(long_liq=$58800.0[$10.0M, -2.00%], short_liq=$61200.0[$10.0M, +2.00%], src=model_est)\n"
    )
    assert audit_rendered_prompt_text(clean_c15) == []

    bad_c15 = (
        "LIVE TELEMETRY SNAPSHOT:\n"
        "### 0. CLOCK & SNAPSHOT: 2026-11-26 17:15:00 EET (Thu)\n"
        "- Unchanged in thread (active from prior turn): Calendar(1/3), Cycle_Anchors(1/3)\n"
        "### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:\n"
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |\n"
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "    | hyperliquid | HYPEUSDC | BUY | 59.5 | 16.8 | 0 | 617202101915 | AI | 4.2h | +0.03% | 99% | 61.4(+3.19%)/58.2(-2.18%) RR@limit=1.46x [ACTIVE] | - |\n"
        "  * Kraken Active Cycle Cost & Drawdown Anchors:\n"
        "    - [kraken] SPCXxUSD: qty=12.0, cost=$1600.00, avg_cost=134.84\n"
        "  * Last Cycle Execution Status: [hyperliquid] HYPEUSDC BUY LIMIT PARTIALLY_FILLED (fill_px=59.5, qty=16.8, id=617202101915, age=0.2m)\n"
        "  * Recent Autonomous Intents:\n"
        "    - [hyperliquid] HYPEUSDC BUY LIMIT FILLED (fill_px=59.5, qty=16.8, id=617202101915, age=0.0h)\n"
        "### 2. MACRO, REGIME & CROSS-MARKET:\n"
        "  * Multi-Month & Annual Structure (3y / 36m lookback):\n"
        "    | Symbol | YTD | 30d | 90d | 52w Range (Pos52w) | 3y Range (Pos) | vs MA50 | vs MA200 | Swings (3y, YY-MM:px) |\n"
        "    |---|---|---|---|---|---|---|---|---|\n"
        "    | BTCUSDC | -31.5% | -25.8% | -23.0% | [$55.0K..$95.0K] (45.0%) | [$40.0K..$120.0K] (70.0%) | -19.8% | -16.8% | 23-11:40.0K->25-10:120.0K->now:62.0K |\n"
        "### 3. TRADABLE MARKETS:\n"
        "  * BTCUSDC: Price=$60,000.00\n"
        "    - Trend: 24h=+1.00% | vol_profile(POC=59500.0[+0.84%], VAL=59000.0[+1.69%], VAH=61000.0[-1.64%], va_pos=12.0%, vp_source=REAL_VOL)\n"
        "    - Book: ob_imbalance=0.500 (bid_depth=$100.0K, ask_depth=$100.0K), spread=0.33bps (bid=59999.0/ask=60001.0), ob_mid=60000.0, "
        "liq_clusters(long_liq=$58800.0[$10.0M, -5.50%], short_liq=$61200.0[$10.0M, +2.00%], src=model_est)\n"
        "  * HYPEUSDC: Price=$59.48\n"
        "    - Trend: 24h=+1.00%\n"
    )
    bad_codes = {v.code for v in audit_rendered_prompt_text(bad_c15)}
    assert {
        "PROMPT_VA_POS_AND_LIQ_MATH",
        "PROMPT_ORDER_LIFECYCLE_CONSISTENCY",
        "PROMPT_WARM_ELISION_TRUTHFULNESS",
        "PROMPT_LONG_HORIZON_MATH",
    }.issubset(bad_codes)








