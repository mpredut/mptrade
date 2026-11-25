"""Regression tests for unified AI intent LIFO windowing, TimedQuote, single-path portfolio normalization, and prompt invariants."""
from __future__ import annotations

import pytest

from botcore import TimedQuote, quote_spread_bps, resolve_aligned_quote_pair
from intelligence.sentiment.collectors.portfolio_collector import derive_runtime_portfolio_view
from intelligence.sentiment.evaluators.microstructure_evaluator import compute_cross_venue_arb_metrics
from providers.ai_attribution import evaluate_ai_attribution
from providers.ai_survival_guard import (
    evaluate_ai_decision_scorecard,
    match_filled_intent_rows_lifo_in_window,
)
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
