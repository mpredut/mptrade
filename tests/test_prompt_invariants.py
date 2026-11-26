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
from verify_tools.audit_prompt_invariants import (
    audit_rendered_prompt_text,
    audit_scorecard_temporal_invariants,
    audit_snapshot_invariants,
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



