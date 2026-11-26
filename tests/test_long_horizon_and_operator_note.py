"""Unit and integration tests for configurable multi-month/annual structure, operator strategic notes, and chart_view zoom."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from botcore import topic_spec
from cachepaths import (
    CACHE_LONG_HORIZON_FILENAME,
    OPERATOR_REGIME_NOTE_FILENAME,
    DurabilityTier,
    cache_ohlc_bars_filename,
    resolve_cachedb_filepath,
    resolve_durability_policy,
)
from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
from intelligence.sentiment.collectors.long_horizon_collector import (
    collect_long_horizon_snapshot,
    load_active_operator_regime_note,
)
from intelligence.sentiment.evaluators.long_horizon_evaluator import (
    compute_zigzag_swings,
    evaluate_symbol_long_horizon,
    format_horizon_label,
    parse_operator_regime_note_text,
    recalculate_long_horizon_with_live_price,
    render_ascii_chart_view,
)
from intelligence.sentiment.evaluators.prompt_delta_evaluator import evaluate_prompt_delta_plan
from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
from orchestratorOS.supervisor.crypto_executor_delta_gate import evaluate_crypto_executor_delta_gate
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
from verify_tools.audit_prompt_invariants import audit_rendered_prompt_text
from providers.ohlc_bar_archive import (
    load_ohlc_archive_bars,
    load_ohlc_bar_archive,
    plan_incremental_bar_fetch,
    upsert_ohlc_bar_archive,
)
from verify_tools.chart_view import main as chart_view_main


def _make_daily_bars(start_date: datetime, prices: List[float]) -> List[List[float]]:
    bars: List[List[float]] = []
    for idx, px in enumerate(prices):
        dt = start_date + timedelta(days=idx)
        ts_ms = dt.timestamp() * 1000.0
        bars.append([ts_ms, px * 0.995, px * 1.015, px * 0.985, px, 1000.0 + idx])
    return bars


def test_long_horizon_configurable_months_zigzag_live_recalc_and_chart_zoom() -> None:
    """Verify configurable month window (18m vs 36m), ZigZag anti-aliasing, MA50/MA200, live recalc, and ASCII zoom."""
    assert format_horizon_label(36) == "3y"
    assert format_horizon_label(24) == "2y"
    assert format_horizon_label(18) == "18m"
    assert format_horizon_label(6) == "6m"

    # Build 420 days of synthetic prices with clear cycle peaks/troughs spanning 2025-10 -> 2026-11
    start_dt = datetime(2025, 10, 1, tzinfo=timezone.utc)
    prices: List[float] = []
    for i in range(420):
        if i < 90:
            prices.append(80000.0 + (i / 90.0) * 30000.0)  # rally 80k -> 110k (through Jan 1 2026 ~110k)
        elif i < 210:
            prices.append(110000.0 - ((i - 90) / 120.0) * 45000.0)  # drop 110k -> 65k
        elif i < 300:
            prices.append(65000.0 + ((i - 210) / 90.0) * 25000.0)  # bounce 65k -> 90k
        else:
            prices.append(90000.0 - ((i - 300) / 119.0) * 30000.0)  # drop 90k -> 60k
    bars = _make_daily_bars(start_dt, prices)
    now_ts = (start_dt + timedelta(days=419)).timestamp()

    swings = compute_zigzag_swings(bars, threshold_pct=15.0, max_pivots=8)
    assert len(swings) >= 3
    assert any(s["kind"] == "H" and s["px"] > 108000.0 for s in swings)
    assert any(s["kind"] == "L" and s["px"] < 66000.0 for s in swings)

    ev_18m = evaluate_symbol_long_horizon("BTCUSDC", bars, months=18, now_ts=now_ts, source="binance:BTCUSDC")
    assert ev_18m is not None
    assert ev_18m["months"] == 18
    assert ev_18m["horizon_label"] == "18m"
    assert ev_18m["ytd_pct"] is not None and ev_18m["ytd_pct"] < -30.0
    assert ev_18m["ma50"] is not None and ev_18m["ma200"] is not None
    assert ev_18m["vs_ma200_pct"] is not None and ev_18m["vs_ma200_pct"] < 0.0
    assert len(ev_18m["weekly_bars"]) >= 50
    assert len(ev_18m["recent_daily_bars"]) == 90

    # Live price recalculation updates percentages and expands range without disk I/O
    recalc = recalculate_long_horizon_with_live_price(ev_18m, 55000.0)
    assert recalc["low_52w"] == 55000.0
    assert recalc["pos_52w_pct"] == 0.0
    assert recalc["ytd_pct"] < ev_18m["ytd_pct"]

    # ASCII chart zoom supports both --months and --from/--to slices cleanly (including day 31 without day-28 clipping)
    chart_txt = render_ascii_chart_view(ev_18m, start_ym="2026-01", end_ym="2026-10", timeframe="1w")
    assert "=== BTCUSDC (1w," in chart_txt
    assert "Swings:" in chart_txt
    chart_oct_1d = render_ascii_chart_view(ev_18m, start_ym="2026-10-01", end_ym="2026-10-31", timeframe="1d")
    assert "2026-10-01 .. 2026-10-31, n=31 bars" in chart_oct_1d
    chart_6m = render_ascii_chart_view(ev_18m, months=6, timeframe="1w")
    assert "=== BTCUSDC (1w," in chart_6m


def test_operator_regime_note_ttl_expiry_zero_tokens_and_delta_gate_wake(tmp_path: Path) -> None:
    """Verify commented/empty/expired operator note adds 0 tokens, active note renders & wakes Delta-Gate, and WARM_DELTA G3 works."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()

    # 1. Comment-only or blank text -> None (zero tokens)
    assert parse_operator_regime_note_text("# only a comment\n\n  # second", mtime_ts=t0, now_ts=t0) is None

    # 2. Active note within default 5-day TTL
    parsed = parse_operator_regime_note_text(
        "# header\nMarket is below the current year's open; respect MA50 resistance.",
        mtime_ts=t0 - 86400.0,
        now_ts=t0,
        default_ttl_days=5.0,
    )
    assert parsed is not None
    assert "below the current year's open" in parsed["text"]
    assert parsed["expires_in_days"] == 4.0

    # 3. Expired after >5 days -> None (zero tokens)
    assert (
        parse_operator_regime_note_text(
            "Old note that must expire",
            mtime_ts=t0 - 5.2 * 86400.0,
            now_ts=t0,
            default_ttl_days=5.0,
        )
        is None
    )

    # 4. Explicit TTL directive overrides default 5d
    parsed_10d = parse_operator_regime_note_text(
        "TTL: 10d\nExtended macro regime directive.",
        mtime_ts=t0 - 7.0 * 86400.0,
        now_ts=t0,
        default_ttl_days=5.0,
    )
    assert parsed_10d is not None and parsed_10d["expires_in_days"] == 3.0

    # 5. Disk loader reads statedb/operator_regime_note.txt via mtime
    statedb_dir = tmp_path / "statedb"
    statedb_dir.mkdir(parents=True, exist_ok=True)
    note_file = statedb_dir / OPERATOR_REGIME_NOTE_FILENAME
    note_file.write_text("# comment\nCrypto is below YTD open; prioritize defense on rallies.\n", encoding="utf-8")
    os.utime(note_file, (t0 - 3600.0, t0 - 3600.0))
    loaded = load_active_operator_regime_note(str(tmp_path), now_ts=t0, default_ttl_days=5.0)
    assert loaded is not None and "below YTD open" in loaded["text"]

    # 6. Delta-Gate wakes immediately when operator note is added/edited
    snap = TelemetrySensorSnapshot(
        evaluated_at_ts=t0,
        prices_and_trends={"BTCUSDC": {"price": 60800.0}},
        operator_regime_note=loaded,
    )
    prev_eval = {
        "ts": t0 - 60.0,
        "delta_gate_meta": {"eval_prices": {"BTCUSDC": 60800.0}, "open_order_ids": [], "plan_states": {}},
    }
    should_skip, reason, _, meta = evaluate_crypto_executor_delta_gate(snap, prev_eval, now_ts=t0)
    assert should_skip is False
    assert reason == "operator_note_updated"
    assert meta.get("operator_note_sig")

    # 7. Clearing or expiring an active note symmetrically wakes Delta-Gate once ('operator_note_cleared'), then stays quiet
    snap_cleared = TelemetrySensorSnapshot(
        evaluated_at_ts=t0 + 60.0,
        prices_and_trends={"BTCUSDC": {"price": 60800.0}},
        operator_regime_note=None,
    )
    prev_with_note = {"ts": t0, "delta_gate_meta": meta}
    skip_clr, rsn_clr, _, meta_clr = evaluate_crypto_executor_delta_gate(snap_cleared, prev_with_note, now_ts=t0 + 60.0)
    assert skip_clr is False
    assert rsn_clr == "operator_note_cleared"
    assert "operator_note_sig" not in meta_clr
    skip_next, _, _, _ = evaluate_crypto_executor_delta_gate(
        snap_cleared, {"ts": t0 + 60.0, "delta_gate_meta": meta_clr}, now_ts=t0 + 120.0
    )
    assert skip_next is True


def test_long_horizon_collector_prompt_rendering_warm_elision_and_chart_cli(tmp_path: Path, capsys: Any) -> None:
    """Verify collector persistence, prompt Section 2 G2/G3 rendering, WARM read-access rule, and chart_view CLI."""
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    start_dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    raw_bn = [
        [
            int((start_dt + timedelta(days=i)).timestamp() * 1000.0),
            str(70000.0 + (i % 60) * 300.0),
            str(71000.0 + (i % 60) * 300.0),
            str(69000.0 + (i % 60) * 300.0),
            str(70500.0 + (i % 60) * 300.0),
            "1500.0",
        ]
        for i in range(420)
    ]

    def _fake_fetch(url: str, timeout: float = 5.0) -> Any:
        _ = timeout
        if "binance.com" in url:
            return raw_bn
        return {}

    cache_dir = str(tmp_path / "cachedb")
    snap_lh = collect_long_horizon_snapshot(
        cache_dir,
        months=18,
        symbols=["BTCUSDC"],
        now_ts=t0,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert snap_lh is not None
    assert snap_lh["months"] == 18
    assert "BTCUSDC" in snap_lh["items"]
    out_file = Path(str(resolve_cachedb_filepath(os.path.join(cache_dir, CACHE_LONG_HORIZON_FILENAME))))
    assert out_file.exists()

    # Verify prompt renders Long_Horizon (G2), Operator_Note (G3), and mptrade/logs read access on both COLD and WARM
    fixed_cal = {"session_label": "EU/US_OVERLAP", "next_high_impact_event": "US Core PCE (2026-11-27 13:30 UTC)"}
    sensors = TelemetrySensorSnapshot(
        evaluated_at_ts=t0,
        macro_calendar=fixed_cal,
        prices_and_trends={"BTCUSDC": {"price": 60800.0, "chg_24h_pct": -1.2, "low_24h": 59500.0, "high_24h": 61800.0}},
        long_horizon=snap_lh,
        operator_regime_note=parse_operator_regime_note_text(
            "Market below current year open.", mtime_ts=t0 - 3600.0, now_ts=t0, default_ttl_days=5.0
        ),
    )
    logs = LogEventSummary(0, 0, 0)
    p1 = evaluate_prompt_delta_plan(sensors, asset_scope="crypto", now_ts=t0, is_warm_resume=False, conversation_id="lh-1")
    cold = build_supervisor_prompt(sensors, logs, spec=topic_spec("auto_cryptotrade_executor"), compact_rules=False, delta_plan=p1)
    assert "FULL READ ACCESS TO `cachedb/` & `statedb/` (`mptrade` workspace)" in cold
    assert "verify_tools/chart_view.py" in cold
    assert "- Operator Strategic Note (updated=" in cold
    assert "Multi-Month & Annual Structure (18m lookback" in cold
    assert "| BTCUSDC |" in cold
    assert audit_rendered_prompt_text(cold) == []

    # Warm turn elides Long_Horizon (1/5) and Operator_Note (unchanged), while preserving mptrade/logs read access
    t2 = t0 + 360.0
    sensors_t2 = TelemetrySensorSnapshot(
        evaluated_at_ts=t2,
        macro_calendar=fixed_cal,
        prices_and_trends={"BTCUSDC": {"price": 60850.0, "chg_24h_pct": -1.1, "low_24h": 59500.0, "high_24h": 61800.0}},
        long_horizon=snap_lh,
        operator_regime_note=parse_operator_regime_note_text(
            "Market below current year open.", mtime_ts=t0 - 3600.0, now_ts=t2, default_ttl_days=5.0
        ),
    )
    p2 = evaluate_prompt_delta_plan(
        sensors_t2, asset_scope="crypto", now_ts=t2, is_warm_resume=True, prev_state=p1.next_state, conversation_id="lh-1"
    )
    warm = build_supervisor_prompt(sensors_t2, logs, spec=topic_spec("auto_cryptotrade_executor"), compact_rules=True, delta_plan=p2)
    assert "`mptrade` read access to `cachedb/`, `statedb/`, `logs/`" in warm
    assert "Long_Horizon(1/5)[src_age=6.0m]" in warm
    assert "Operator_Note(unchanged)" in warm
    assert "Multi-Month & Annual Structure" not in warm
    assert "- Operator Strategic Note" not in warm
    assert audit_rendered_prompt_text(warm) == []

    # When operator note expires or is cleared on Turn 3, WARM_DELTA notifies ONCE, then adds 0 tokens on Turn 4
    t3 = t2 + 360.0
    sensors_t3 = TelemetrySensorSnapshot(
        evaluated_at_ts=t3,
        macro_calendar=fixed_cal,
        prices_and_trends={"BTCUSDC": {"price": 60860.0, "chg_24h_pct": -1.1, "low_24h": 59500.0, "high_24h": 61800.0}},
        long_horizon=snap_lh,
        operator_regime_note=None,
    )
    p3 = evaluate_prompt_delta_plan(
        sensors_t3, asset_scope="crypto", now_ts=t3, is_warm_resume=True, prev_state=p2.next_state, conversation_id="lh-1"
    )
    warm_clr = build_supervisor_prompt(sensors_t3, logs, spec=topic_spec("auto_cryptotrade_executor"), compact_rules=True, delta_plan=p3)
    assert "Operator_Note(EXPIRED_CLEARED)" in warm_clr
    assert "- Operator Strategic Note" not in warm_clr
    assert audit_rendered_prompt_text(warm_clr) == []

    t4 = t3 + 360.0
    p4 = evaluate_prompt_delta_plan(
        sensors_t3, asset_scope="crypto", now_ts=t4, is_warm_resume=True, prev_state=p3.next_state, conversation_id="lh-1"
    )
    warm_after = build_supervisor_prompt(sensors_t3, logs, spec=topic_spec("auto_cryptotrade_executor"), compact_rules=True, delta_plan=p4)
    assert "Operator_Note" not in warm_after
    assert audit_rendered_prompt_text(warm_after) == []

    # Verify CLI chart_view works end-to-end on persisted snapshot and full multi-year 1d archive
    assert "weekly_bars" not in snap_lh["items"]["BTCUSDC"]
    assert "recent_daily_bars" not in snap_lh["items"]["BTCUSDC"]
    assert snap_lh["items"]["BTCUSDC"]["bar_archive"] == cache_ohlc_bars_filename("BTCUSDC", "1d")

    rc = chart_view_main(["BTCUSDC", "--months", "12", "--cache-dir", cache_dir])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "=== BTCUSDC (1w," in captured

    # Full multi-year daily zoom (--tf 1d) reads directly from cache_ohlc_bars_1d_BTCUSDC.json (>90d ago!)
    rc_1d = chart_view_main(["BTCUSDC", "--tf", "1d", "--from", "2025-07", "--to", "2025-09", "--cache-dir", cache_dir])
    assert rc_1d == 0
    cap_1d = capsys.readouterr().out
    assert "=== BTCUSDC (1d, 2025-07-01 .. 2025-09-30" in cap_1d


def test_cycle2_unactionable_wall_trapped_suppression_equities_live_lh_and_cycle_anchors_g1() -> None:
    """Verify Cycle 2 audit fixes:
    1. Delta-Gate and Section 0 Trigger ignore unactionable wall_trapped front-runs whose RR < min_target_rr,
       while still waking/promoting actionable front-runs (RR >= min_target_rr) and unlinked orders.
    2. Equities-scope Long_Horizon live-recalculates BTCUSDC & SPY from cross-market/benchmarks live prices.
    3. Active Cycle Cost & Drawdown Anchors (venue_cycles) elide cleanly under G1 on WARM turns and wake on fill changes,
       while WARM turns explicitly retain verify_tools/chart_view.py read access.
    """
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=ZoneInfo("Europe/Bucharest")).timestamp()
    # 1. Delta-Gate suppresses unactionable BTCUSDC front_run (RR=0.41x < 0.5x)
    base_ps: Dict[str, Any] = {
        "open_limit_orders": [
            {
                "venue": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "order_id": "10976647800",
                "price": 59680.0,
                "qty": 0.015,
                "filled_qty": 0.0,
                "status": "NEW",
                "dist_from_spot_pct": -0.95,
                "fill_prob_1h_pct": 44.0,
                "recommended_front_run_px": 60164.62,
            }
        ],
        "active_ai_trade_plans": [
            {
                "venue": "binance",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "order_state": "RESTING",
                "entry_px": 59680.0,
                "target_px": 60850.0,
                "invalidation_px": 58500.0,
                "status": "ACTIVE",
            }
        ],
    }
    snap_bad_rr = TelemetrySensorSnapshot(
        evaluated_at_ts=t0 + 360.0,
        prices_and_trends={"BTCUSDC": {"price": 60250.0}},
        portfolio_summary=base_ps,
    )
    prev_eval = {
        "ts": t0,
        "delta_gate_meta": {
            "eval_prices": {"BTCUSDC": 60250.0},
            "open_order_ids": ["10976647800"],
            "order_attention_flags": {},
            "plan_states": {},
        },
    }
    skip_bad, rsn_bad, _, meta_bad = evaluate_crypto_executor_delta_gate(
        snap_bad_rr, prev_eval, now_ts=t0 + 360.0, min_target_rr=0.5
    )
    assert skip_bad is True and rsn_bad.startswith("quiet_flat")
    assert meta_bad["order_attention_flags"] == {}

    # Adding an actionable HYPEUSDC front_run (RR=1.45x >= 0.5x) wakes Delta-Gate AND Trigger highlights ONLY HYPEUSDC
    ps_mixed = {
        "open_limit_orders": [
            *base_ps["open_limit_orders"],
            {
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "side": "BUY",
                "order_id": "617202101915",
                "price": 59.5,
                "qty": 16.8,
                "filled_qty": 0.0,
                "status": "NEW",
                "dist_from_spot_pct": -0.58,
                "fill_prob_1h_pct": 80.0,
                "recommended_front_run_px": 59.506,
            },
        ],
        "active_ai_trade_plans": [
            *base_ps["active_ai_trade_plans"],
            {
                "venue": "hyperliquid",
                "symbol": "HYPEUSDC",
                "side": "BUY",
                "order_state": "RESTING",
                "entry_px": 59.5,
                "target_px": 61.4,
                "invalidation_px": 58.2,
                "status": "ACTIVE",
            },
        ],
    }
    snap_mixed = TelemetrySensorSnapshot(
        evaluated_at_ts=t0 + 360.0,
        prices_and_trends={"BTCUSDC": {"price": 60250.0}, "HYPEUSDC": {"price": 59.85}},
        portfolio_summary=ps_mixed,
    )
    prev_mixed = {
        "ts": t0,
        "delta_gate_meta": {
            "eval_prices": {"BTCUSDC": 60250.0, "HYPEUSDC": 59.85},
            "open_order_ids": ["10976647800", "617202101915"],
            "order_attention_flags": {},
            "plan_states": {},
        },
    }
    skip_ok, rsn_ok, _, meta_ok = evaluate_crypto_executor_delta_gate(
        snap_mixed, prev_mixed, now_ts=t0 + 360.0, min_target_rr=0.5
    )
    assert (skip_ok, rsn_ok) == (False, "open_order_wall_trapped")
    assert meta_ok["order_attention_flags"] == {"617202101915": "open_order_wall_trapped"}
    trig_prompt = build_supervisor_prompt(
        snap_mixed,
        LogEventSummary(0, 0, 0),
        spec=topic_spec("auto_cryptotrade_executor"),
        compact_rules=False,
        wake_reason="open_order_wall_trapped",
    )
    assert "- Trigger: open_order_wall_trapped | HYPEUSDC wall_trapped(front_run=59.506,RR=1.45x)" in trig_prompt
    assert "BTCUSDC wall_trapped" not in trig_prompt
    s3_txt = trig_prompt[trig_prompt.index("### 3. TRADABLE MARKETS"):]
    assert s3_txt.index("  * HYPEUSDC:") < s3_txt.index("  * BTCUSDC:")

    # 2. Equities-scope Long_Horizon live-recalculates BTCUSDC & SPY from cross_market_context & macro_benchmarks
    # 3. Active Cycle Cost & Drawdown Anchors elide cleanly on WARM turns under G1 and return [NEW] on fill change
    lh_eq_snap = {
        "ts": t0 - 1800.0,
        "months": 36,
        "horizon_label": "3y",
        "items": {
            "BTCUSDC": {
                "symbol": "BTCUSDC",
                "last_close": 60800.0,
                "ytd_ref": 87500.0,
                "ytd_pct": -30.51,
                "low_52w": 55100.0,
                "high_52w": 98000.0,
                "pos_52w_pct": 13.3,
                "low_horizon": 36700.0,
                "high_horizon": 126300.0,
                "pos_horizon_pct": 26.9,
                "ma50": 74700.0,
                "vs_ma50_pct": -18.61,
                "ma200": 71900.0,
                "vs_ma200_pct": -15.44,
                "swings": [{"ym": "25-10", "px": 126300.0, "kind": "H"}, {"ym": "26-11", "px": 55100.0, "kind": "L"}],
            },
            "SPY": {
                "symbol": "SPY",
                "last_close": 716.7,
                "ytd_ref": 682.0,
                "ytd_pct": 5.09,
                "low_52w": 629.0,
                "high_52w": 782.0,
                "pos_52w_pct": 57.3,
                "low_horizon": 453.0,
                "high_horizon": 782.0,
                "pos_horizon_pct": 80.2,
                "ma50": 758.0,
                "vs_ma50_pct": -5.45,
                "ma200": 733.0,
                "vs_ma200_pct": -2.22,
                "swings": [{"ym": "26-10", "px": 782.0, "kind": "H"}],
            },
        },
    }
    eq_ps = {
        "cross_market_context": {"crypto_bellwethers": {"BTC": {"price": 58000.0, "chg_24h_pct": -2.0}}},
        "t212_cycles": {
            "NVDA_US_EQ": {
                "qty": 1.69,
                "cost_usd": 249.22,
                "avg_cost_px": 147.47,
                "entry_price": 147.47,
                "last_buy_price": 147.47,
                "last_sell_price": 156.15,
                "dca_buys": 0,
                "pnl_vs_avg_pct": 2.6,
            }
        },
    }
    eq_sensors = TelemetrySensorSnapshot(
        evaluated_at_ts=t0,
        macro_benchmarks={"items": {"SPY": {"price": 730.0}}, "summary_line": "SPY=$730.00"},
        macro_calendar={"session_label": "US_RTH", "next_high_impact_event": "FOMC"},
        prices_and_trends={"NVDA_US_EQ": {"price": 151.3, "chg_24h_pct": 1.0, "low_24h": 148.0, "high_24h": 152.0}},
        long_horizon=lh_eq_snap,
        portfolio_summary=eq_ps,
    )
    eq_p1 = evaluate_prompt_delta_plan(eq_sensors, asset_scope="equities", now_ts=t0, is_warm_resume=False, conversation_id="eq-c2")
    eq_cold = build_supervisor_prompt(eq_sensors, LogEventSummary(0, 0, 0), spec=topic_spec("auto_sharetrade_executor"), compact_rules=False, delta_plan=eq_p1)
    # BTCUSDC recalculated live to 58.0K (not stale 60.8K), SPY recalculated live to 730.0 (not stale 716.7)
    assert "now:58.0K" in eq_cold and "now:60.8K" not in eq_cold
    assert "now:730.0" in eq_cold and "now:716.7" not in eq_cold
    assert "Trading212 Active Cycle Cost & Drawdown Anchors:" in eq_cold

    # Warm turn elides unchanged Cycle_Anchors (even when live pnl_vs_avg_pct ticks) and includes chart_view.py in rules
    eq_ps_t2 = {
        **eq_ps,
        "t212_cycles": {"NVDA_US_EQ": {**eq_ps["t212_cycles"]["NVDA_US_EQ"], "pnl_vs_avg_pct": 3.1}},
    }
    eq_sensors_t2 = TelemetrySensorSnapshot(
        evaluated_at_ts=t0 + 360.0,
        macro_benchmarks=eq_sensors.macro_benchmarks,
        macro_calendar=eq_sensors.macro_calendar,
        prices_and_trends={"NVDA_US_EQ": {"price": 152.0, "chg_24h_pct": 1.5, "low_24h": 148.0, "high_24h": 152.5}},
        long_horizon=lh_eq_snap,
        portfolio_summary=eq_ps_t2,
    )
    eq_p2 = evaluate_prompt_delta_plan(
        eq_sensors_t2, asset_scope="equities", now_ts=t0 + 360.0, is_warm_resume=True, prev_state=eq_p1.next_state, conversation_id="eq-c2"
    )
    eq_warm = build_supervisor_prompt(eq_sensors_t2, LogEventSummary(0, 0, 0), spec=topic_spec("auto_sharetrade_executor"), compact_rules=True, delta_plan=eq_p2)
    assert "Cycle_Anchors(1/3)" in eq_warm
    assert "Active Cycle Cost & Drawdown Anchors" not in eq_warm
    assert "verify_tools/chart_view.py" in eq_warm

    # Fill changing qty/dca_buys immediately shows Active Cycle Cost & Drawdown Anchors [NEW]
    eq_ps_t3 = {
        **eq_ps,
        "t212_cycles": {"NVDA_US_EQ": {**eq_ps["t212_cycles"]["NVDA_US_EQ"], "qty": 2.69, "dca_buys": 1}},
    }
    eq_sensors_t3 = TelemetrySensorSnapshot(
        evaluated_at_ts=t0 + 720.0,
        macro_benchmarks=eq_sensors.macro_benchmarks,
        macro_calendar=eq_sensors.macro_calendar,
        prices_and_trends=eq_sensors_t2.prices_and_trends,
        long_horizon=lh_eq_snap,
        portfolio_summary=eq_ps_t3,
    )
    eq_p3 = evaluate_prompt_delta_plan(
        eq_sensors_t3, asset_scope="equities", now_ts=t0 + 720.0, is_warm_resume=True, prev_state=eq_p2.next_state, conversation_id="eq-c2"
    )
    eq_warm_new = build_supervisor_prompt(eq_sensors_t3, LogEventSummary(0, 0, 0), spec=topic_spec("auto_sharetrade_executor"), compact_rules=True, delta_plan=eq_p3)
    assert "Trading212 Active Cycle Cost & Drawdown Anchors [NEW]:" in eq_warm_new


def test_persistent_ohlc_bar_archive_incremental_refresh_extend_kraken_and_multi_tf(tmp_path: Path) -> None:
    """Verify durable OHLCV archive never re-fetches persisted history, extends on larger horizon,
    accumulates Kraken past 720d, resists offline fallback downgrade, and is multi-timeframe ready."""
    from urllib.parse import parse_qs, urlparse

    cache_dir = str(tmp_path / "cachedb")
    os.makedirs(os.path.join(cache_dir, "prices"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "macro"), exist_ok=True)

    # 1. SSOT path naming & ARCHIVE_DURABLE policy (incl. mixed-case xStock & sub-daily Option 3 prep)
    assert cache_ohlc_bars_filename("BTCUSDC", "1d") == "cache_ohlc_bars_1d_BTCUSDC.json"
    assert cache_ohlc_bars_filename("SPCXxUSD", "1d") == "cache_ohlc_bars_1d_SPCXxUSD.json"
    assert cache_ohlc_bars_filename("BTCUSDC", "4h") == "cache_ohlc_bars_4h_BTCUSDC.json"
    assert resolve_durability_policy(cache_ohlc_bars_filename("BTCUSDC", "1d")).tier is DurabilityTier.ARCHIVE_DURABLE

    # 2. Full initial backfill for 18m (Binance BTCUSDC + Yahoo NVDA_US_EQ), then incremental 6h refresh
    t0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    start_3y = datetime(2023, 5, 1, tzinfo=timezone.utc)
    n_seed_days = (datetime(2026, 11, 20, tzinfo=timezone.utc) - start_3y).days + 1
    all_bn_rows = [
        [
            int((start_3y + timedelta(days=i)).timestamp() * 1000.0),
            f"{50000.0 + i * 20.0:.2f}",
            f"{50500.0 + i * 20.0:.2f}",
            f"{49500.0 + i * 20.0:.2f}",
            f"{50200.0 + i * 20.0:.2f}",
            "1200.0",
        ]
        for i in range(n_seed_days)
    ]
    bn_calls: List[Dict[str, int]] = []
    yahoo_ranges: List[str] = []

    def _fake_fetch(url: str, timeout: float = 5.0) -> Any:
        _ = timeout
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "binance.com" in url:
            lim = int(qs["limit"][0])
            end_ms = int(qs["endTime"][0])
            bn_calls.append({"limit": lim, "endTime": end_ms})
            eligible = [r for r in all_bn_rows if r[0] <= end_ms]
            return eligible[-lim:]
        if "finance.yahoo.com" in url:
            rng = qs["range"][0]
            yahoo_ranges.append(rng)
            n_bars = 22 if rng == "1mo" else 600
            base_dt = datetime(2026, 11, 20, tzinfo=timezone.utc) - timedelta(days=n_bars)
            ts_list = [int((base_dt + timedelta(days=i)).timestamp()) for i in range(n_bars)]
            closes = [120.0 + i * 0.1 for i in range(n_bars)]
            return {
                "chart": {
                    "result": [{
                        "meta": {"symbol": "NVDA"},
                        "timestamp": ts_list,
                        "indicators": {"quote": [{"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000.0] * n_bars}]},
                    }]
                }
            }
        return {}

    snap1 = collect_long_horizon_snapshot(
        cache_dir,
        months=18,
        symbols=["BTCUSDC", "NVDA_US_EQ"],
        now_ts=t0,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert snap1 is not None
    assert len(bn_calls) == 1 and bn_calls[0]["limit"] >= 700
    assert yahoo_ranges == ["5y"]
    arch_btc_1 = load_ohlc_bar_archive(cache_dir, "BTCUSDC", "1d")
    assert arch_btc_1 is not None and arch_btc_1["bar_count"] >= 700

    # Second run 1 day later -> strictly incremental tail fetch (no full re-fetch, no volume double-counting!)
    bn_calls.clear()
    yahoo_ranges.clear()
    t1 = t0 + 86400.0
    extra_day_dt = start_3y + timedelta(days=len(all_bn_rows))
    all_bn_rows.append([int(extra_day_dt.timestamp() * 1000.0), "76500.0", "77000.0", "76000.0", "76800.0", "1500.0"])
    snap2 = collect_long_horizon_snapshot(
        cache_dir,
        months=18,
        symbols=["BTCUSDC", "NVDA_US_EQ"],
        now_ts=t1,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert snap2 is not None
    assert len(bn_calls) == 1 and bn_calls[0]["limit"] <= 12
    assert yahoo_ranges == ["1mo"]
    arch_btc_2 = load_ohlc_bar_archive(cache_dir, "BTCUSDC", "1d")
    assert arch_btc_2 is not None and arch_btc_2["bar_count"] == arch_btc_1["bar_count"] + 1
    # Overlapping day volume was replaced, NOT summed (1200.0, not 2400.0)
    assert arch_btc_2["bars"][-2][5] == 1200.0
    assert arch_btc_2["bars"][-1][5] == 1500.0

    # 3. Expanding horizon (18m -> 36m) fetches ONLY older prefix + tiny tail, never re-fetches persisted middle
    bn_calls.clear()
    snap3 = collect_long_horizon_snapshot(
        cache_dir,
        months=36,
        symbols=["BTCUSDC"],
        now_ts=t1,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert snap3 is not None and snap3["months"] == 36
    assert len(bn_calls) == 2
    assert bn_calls[0]["endTime"] < arch_btc_2["first_ts_ms"]
    assert bn_calls[1]["limit"] <= 12
    arch_btc_3 = load_ohlc_bar_archive(cache_dir, "BTCUSDC", "1d")
    assert arch_btc_3 is not None and arch_btc_3["bar_count"] == len(all_bn_rows)

    # Next 36m run returns immediately to tiny incremental fetch
    bn_calls.clear()
    collect_long_horizon_snapshot(
        cache_dir,
        months=36,
        symbols=["BTCUSDC"],
        now_ts=t1 + 6 * 3600.0,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert len(bn_calls) == 1 and bn_calls[0]["limit"] <= 12

    # 4. Short-history asset & Kraken >720d persistence + offline fallback protection
    kraken_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    kraken_seed = _make_daily_bars(kraken_start, [20.0 + i * 0.05 for i in range(720)])
    t_k = (kraken_start + timedelta(days=750)).timestamp()
    upsert_ohlc_bar_archive(
        cache_dir,
        "TAOUSD",
        kraken_seed,
        timeframe="1d",
        source="kraken:TAOUSD",
        requested_from_ms=int((kraken_start - timedelta(days=365)).timestamp() * 1000),
        now_ts=t_k,
    )
    # Subsequent incremental fetch adds newer days without losing older bars -> >720 days preserved
    newer_kraken = _make_daily_bars(kraken_start + timedelta(days=718), [60.0 + i for i in range(32)])
    merged_k = upsert_ohlc_bar_archive(
        cache_dir,
        "TAOUSD",
        newer_kraken,
        timeframe="1d",
        source="kraken:TAOUSD",
        requested_from_ms=int((kraken_start - timedelta(days=365)).timestamp() * 1000),
        now_ts=t_k,
    )
    assert merged_k is not None and merged_k["bar_count"] == 750
    plan_k = plan_incremental_bar_fetch(
        merged_k,
        needed_start_ms=int((kraken_start - timedelta(days=300)).timestamp() * 1000),
        now_ms=int((t_k + 86400.0) * 1000),
        timeframe="1d",
    )
    assert plan_k.mode == "incremental"

    # Offline local_sparse fallback NEVER overwrites authoritative exchange OHLCV bars
    sparse_override = [[merged_k["bars"][-1][0], 1.0]]
    after_fb = upsert_ohlc_bar_archive(
        cache_dir,
        "TAOUSD",
        sparse_override,
        timeframe="1d",
        source="local_sparse",
        incoming_is_fallback=True,
        now_ts=t_k,
    )
    assert after_fb is not None
    assert after_fb["source"] == "kraken:TAOUSD"
    assert after_fb["bars"][-1][4] == merged_k["bars"][-1][4]

    # 5. Option 3 multi-timeframe readiness: 4h archive coexists independently from 1d archive
    t_4h_base = int(datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    bars_4h = [[t_4h_base + i * 4 * 3600 * 1000, 60000.0, 60500.0, 59800.0, 60200.0 + i * 10, 250.0] for i in range(12)]
    doc_4h = upsert_ohlc_bar_archive(
        cache_dir,
        "BTCUSDC",
        bars_4h,
        timeframe="4h",
        source="binance:BTCUSDC",
        requested_from_ms=t_4h_base,
        now_ts=t_4h_base / 1000.0 + 48 * 3600,
    )
    assert doc_4h is not None and doc_4h["timeframe"] == "4h" and doc_4h["interval_min"] == 240
    assert len(load_ohlc_archive_bars(cache_dir, "BTCUSDC", "4h")) == 12
    assert len(load_ohlc_archive_bars(cache_dir, "BTCUSDC", "1d")) == len(all_bn_rows)
    plan_4h = plan_incremental_bar_fetch(
        doc_4h,
        needed_start_ms=t_4h_base,
        now_ms=t_4h_base + 52 * 3600 * 1000,
        timeframe="4h",
        overlap_bars=3,
    )
    assert plan_4h.mode == "incremental"
    assert plan_4h.interval_min == 240


def test_cycle3_secondary_sibling_impulse_suppression_and_legacy_archive_bootstrap(tmp_path: Path) -> None:
    """Verify Cycle 3 fixes:
    1. evaluate_kinematic_impulse_symbols suppresses thin-book secondary sibling (HYPEUSD) spikes when
       primary deep-book venue (HYPEUSDC) carries live calm kinematics, while keeping genuine shocks and
       sole-quote symbols intact.
    2. LongHorizonStructureCollector bypasses skip_if_fresh_sec when a legacy fresh snapshot lacks
       persistent cache_ohlc_bars_1d_<SYM>.json archive files on disk, then skips normally once seeded.
    """
    from intelligence.globaltelemetry_collector import evaluate_kinematic_impulse_symbols
    from state_io import atomic_write_json

    # 1. Secondary sibling HYPEUSD spike suppressed when primary HYPEUSDC is calm
    assert evaluate_kinematic_impulse_symbols(
        {
            "HYPEUSDC": {"price": 60.10, "chg_3m_pct": 0.12, "vel_90s_pct_min": 0.05, "accel_90s_pct_min2": 0.02},
            "HYPEUSD": {"price": 60.40, "chg_3m_pct": 0.85, "vel_90s_pct_min": 0.45, "accel_90s_pct_min2": 0.30},
        }
    ) == []
    # When primary HYPEUSDC genuinely shocks, it triggers cleanly
    assert evaluate_kinematic_impulse_symbols(
        {
            "HYPEUSDC": {"price": 60.80, "chg_3m_pct": 0.82, "vel_90s_pct_min": 0.42, "accel_90s_pct_min2": 0.25},
            "HYPEUSD": {"price": 60.75, "chg_3m_pct": 0.80, "vel_90s_pct_min": 0.40, "accel_90s_pct_min2": 0.24},
        }
    ) == ["HYPEUSDC"]
    # When HYPEUSD is the ONLY live symbol for base HYPE, it still triggers cleanly
    assert evaluate_kinematic_impulse_symbols(
        {
            "HYPEUSD": {"price": 60.40, "chg_3m_pct": 0.85, "vel_90s_pct_min": 0.45, "accel_90s_pct_min2": 0.30},
        }
    ) == ["HYPEUSD"]

    # 2. Legacy fresh long_horizon_collect.json without bar_archive triggers immediate archive bootstrap
    cache_dir = str(tmp_path / "cachedb")
    t0 = datetime(2026, 11, 26, 15, 0, 0, tzinfo=timezone.utc).timestamp()
    legacy_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, CACHE_LONG_HORIZON_FILENAME)))
    atomic_write_json(
        legacy_path,
        {
            "ts": t0 - 60.0,
            "months": 12,
            "horizon_label": "1y",
            "symbol_count": 1,
            "items": {"BTCUSDC": {"symbol": "BTCUSDC", "last_close": 60000.0}},
        },
    )
    start_dt = datetime(2025, 11, 1, tzinfo=timezone.utc)
    bn_rows = [
        [
            int((start_dt + timedelta(days=i)).timestamp() * 1000.0),
            "60000.0",
            "60500.0",
            "59500.0",
            str(60000.0 + i * 5.0),
            "1000.0",
        ]
        for i in range(40)
    ]
    calls = {"n": 0}

    def _fake_fetch(url: str, timeout: float = 5.0) -> Any:
        _ = (url, timeout)
        calls["n"] += 1
        return bn_rows

    boot = collect_long_horizon_snapshot(
        cache_dir,
        months=12,
        symbols=["BTCUSDC"],
        now_ts=t0,
        persist=True,
        skip_if_fresh_sec=21600.0,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert boot is not None and calls["n"] >= 1
    assert boot["items"]["BTCUSDC"].get("bar_archive") == cache_ohlc_bars_filename("BTCUSDC", "1d")
    assert load_ohlc_bar_archive(cache_dir, "BTCUSDC", "1d") is not None

    # Subsequent call within skip_if_fresh_sec now skips cleanly with 0 network calls
    calls_before = calls["n"]
    cached = collect_long_horizon_snapshot(
        cache_dir,
        months=12,
        symbols=["BTCUSDC"],
        now_ts=t0 + 120.0,
        persist=True,
        skip_if_fresh_sec=21600.0,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert cached is not None and calls["n"] == calls_before


def test_option3_unified_multi_tf_intraday_collector_market_api_consumers_and_audit_fixes(
    tmp_path: Path, capsys: Any, monkeypatch: Any
) -> None:
    """Verify Option 2 audit fixes & Option 3 unified multi-TF archive (1h/4h/1d):
    1. Kraken tokenized xStocks pass asset_class=tokenized_asset.
    2. Intraday collector persists 1h & 4h for crypto incrementally, and fetches 1h + resamples 4h locally for Yahoo equities.
    3. MarketApi.fetch_historical_ohlc_points (survival) and fetch_ohlc_candles (walk-forward) serve fresh archives with 0 REST calls,
       while falling back to REST when a crypto archive has a cadence gap.
    4. chart_view CLI supports --tf 4h and --tf 1h directly from persisted archives, and retention pruning bounds 1h length.
    """
    from urllib.parse import parse_qs, urlparse
    from intelligence.sentiment.collectors.long_horizon_collector import collect_intraday_ohlc_archives
    from providers.market_api import api as mkt
    from providers.ohlc_bar_archive import filter_closed_archive_bars, load_ohlc_archive_candles

    cache_dir = str(tmp_path / "cachedb")
    os.makedirs(os.path.join(cache_dir, "prices"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "macro"), exist_ok=True)

    # Anchor now_ts right on a closed 4h boundary so closed-bar checks are deterministic
    now_dt = datetime(2026, 11, 26, 16, 0, 0, tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()
    now_ms = int(now_ts * 1000)

    # Build contiguous 1h bars (120h = 5d) and 4h bars (60 bars = 10d) for BTCUSDC, plus 1h Yahoo bars for NVDA_US_EQ
    start_1h_ms = now_ms - 120 * 3600 * 1000
    bn_1h = [
        [start_1h_ms + i * 3600 * 1000, f"{60000 + i:.2f}", f"{60200 + i:.2f}", f"{59800 + i:.2f}", f"{60100 + i:.2f}", "100.0"]
        for i in range(121)  # includes forming bar at now_ms
    ]
    start_4h_ms = now_ms - 60 * 4 * 3600 * 1000
    bn_4h = [
        [start_4h_ms + i * 4 * 3600 * 1000, f"{59500 + i * 5:.2f}", f"{59900 + i * 5:.2f}", f"{59300 + i * 5:.2f}", f"{59700 + i * 5:.2f}", "400.0"]
        for i in range(61)  # includes forming bar at now_ms
    ]
    krk_urls: List[str] = []
    yahoo_calls: List[Tuple[str, str]] = []
    bn_calls: List[Tuple[str, int]] = []

    def _fake_fetch(url: str, timeout: float = 5.0) -> Any:
        _ = timeout
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "binance.com" in url:
            iv = qs["interval"][0]
            lim = int(qs["limit"][0])
            end_ms = int(qs["endTime"][0])
            bn_calls.append((iv, lim))
            src = bn_1h if iv == "1h" else bn_4h
            eligible = [r for r in src if r[0] <= end_ms]
            return eligible[-lim:]
        if "kraken.com" in url:
            krk_urls.append(url)
            iv_m = int(qs["interval"][0])
            step_s = iv_m * 60
            return {
                "error": [],
                "result": {
                    "SPCXxUSD": [
                        [int(now_ts) - (40 - i) * step_s, "140.0", "142.0", "139.0", f"{140.5 + i * 0.1:.2f}", "140.5", "50.0", 10]
                        for i in range(40)
                    ],
                    "last": int(now_ts),
                },
            }
        if "finance.yahoo.com" in url:
            rng = qs["range"][0]
            iv = qs["interval"][0]
            yahoo_calls.append((iv, rng))
            ts_list = [int((start_1h_ms + i * 3600 * 1000) // 1000) for i in range(48)]
            closes = [150.0 + i * 0.25 for i in range(48)]
            return {
                "chart": {
                    "result": [{
                        "meta": {"symbol": "NVDA"},
                        "timestamp": ts_list,
                        "indicators": {"quote": [{"open": closes, "high": closes, "low": closes, "close": closes, "volume": [500.0] * 48}]},
                    }]
                }
            }
        return {}

    # 1. Cold backfill via IntradayOhlcArchiveCollector (BTCUSDC + NVDA_US_EQ + SPCXxUSD)
    res1 = collect_intraday_ohlc_archives(
        cache_dir,
        symbols=["BTCUSDC", "NVDA_US_EQ", "SPCXxUSD"],
        now_ts=now_ts,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert res1 is not None and res1["symbol_count"] == 3
    # Audit Fix 1: Kraken tokenized xStock includes asset_class=tokenized_asset
    assert krk_urls and all("asset_class=tokenized_asset" in u for u in krk_urls)
    # Yahoo equities fetched ONLY 1h once (4h derived locally with ZERO extra Yahoo calls!)
    assert yahoo_calls == [("1h", "1y")]
    nvda_1h = load_ohlc_bar_archive(cache_dir, "NVDA_US_EQ", "1h")
    nvda_4h = load_ohlc_bar_archive(cache_dir, "NVDA_US_EQ", "4h")
    assert nvda_1h is not None and nvda_1h["bar_count"] == 48
    assert nvda_4h is not None and nvda_4h["bar_count"] == 12 and "resampled_1h" in nvda_4h["source"]
    # Verify Yahoo timestamps were normalized to ms (Audit Fix 4)
    assert nvda_1h["bars"][0][0] > 1e11

    # 2. Fresh-guard skip within 1h -> 0 HTTP calls
    bn_calls.clear()
    yahoo_calls.clear()
    res_skip = collect_intraday_ohlc_archives(
        cache_dir,
        symbols=["BTCUSDC", "NVDA_US_EQ", "SPCXxUSD"],
        now_ts=now_ts + 600.0,
        persist=True,
        skip_if_fresh_sec=3600.0,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert res_skip is not None and res_skip.get("skipped_fresh") is True
    assert bn_calls == [] and yahoo_calls == []

    # 3. Incremental refresh 2h later -> only tiny tail fetched (Binance limit <= 16, Yahoo range=5d!)
    bn_calls.clear()
    yahoo_calls.clear()
    t_inc = now_ts + 2 * 3600.0
    bn_1h.append([now_ms + 3600 * 1000, "60200.0", "60400.0", "60100.0", "60300.0", "110.0"])
    bn_1h.append([now_ms + 2 * 3600 * 1000, "60300.0", "60500.0", "60200.0", "60400.0", "120.0"])
    res2 = collect_intraday_ohlc_archives(
        cache_dir,
        symbols=["BTCUSDC", "NVDA_US_EQ"],
        now_ts=t_inc,
        persist=True,
        fetch_fn=_fake_fetch,
        workspace_dir=str(tmp_path),
    )
    assert res2 is not None
    assert all(lim <= 20 for _, lim in bn_calls)
    assert yahoo_calls == [("1h", "5d")]

    # 4. Closed-bar filtering drops forming bar when drop_incomplete=True
    btc_1h_bars = load_ohlc_archive_bars(cache_dir, "BTCUSDC", "1h")
    closed_1h = filter_closed_archive_bars(btc_1h_bars, interval_min=60, now_ms=int(t_inc * 1000))
    assert len(closed_1h) == len(btc_1h_bars) - 1
    candles_1h = load_ohlc_archive_candles(cache_dir, "BTCUSDC", "1h", drop_incomplete=True, now_ms=int(t_inc * 1000))
    assert len(candles_1h) == len(closed_1h) and set(candles_1h[0].keys()) >= {"timestamp", "open", "high", "low", "close", "volume"}

    # 5. Option 3 consumers: MarketApi serves archive-first with ZERO live provider calls
    monkeypatch.setattr("providers.ohlc_bar_archive.resolve_now_ts", lambda now_ts=None: t_inc if now_ts is None else now_ts)
    for prov in mkt._providers:
        monkeypatch.setattr(prov, "fetch_historical_ohlc_points", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected REST points fetch")))
        monkeypatch.setattr(prov, "fetch_ohlc_candles", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected REST candles fetch")))

    pts_1h = mkt.fetch_historical_ohlc_points("BTCUSDC", days=3, interval="1h", cache_dir=cache_dir)
    assert len(pts_1h) >= 70
    ds_4h = mkt.fetch_ohlc_candles("BTCUSDC", interval=240, lookback_days=5, drop_incomplete=True, cache_dir=cache_dir)
    assert len(ds_4h) >= 25
    # Verify contiguity fallback: introduce a 2-hour gap -> MarketApi.fetch_ohlc_candles rejects gapped archive & hits provider
    gapped_bars = [b for idx, b in enumerate(btc_1h_bars) if idx != len(btc_1h_bars) - 5]
    upsert_ohlc_bar_archive(cache_dir, "ETHUSDC", gapped_bars, timeframe="1h", source="binance:ETHUSDC", requested_from_ms=start_1h_ms, now_ts=t_inc)
    rest_fallback_hit = {"called": False}
    for prov in mkt._providers:
        monkeypatch.setattr(
            prov,
            "fetch_ohlc_candles",
            lambda *a, **k: (rest_fallback_hit.__setitem__("called", True) or [{"timestamp": int(t_inc) - 3600, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}]),
        )
    fb_res = mkt.fetch_ohlc_candles("ETHUSDC", interval=60, lookback_days=2, drop_incomplete=True, cache_dir=cache_dir)
    assert rest_fallback_hit["called"] is True and len(fb_res) == 1

    # 6. Retention pruning bounds 1h archive cleanly
    old_bar = [int((now_dt - timedelta(days=400)).timestamp() * 1000), 10000.0, 10100.0, 9900.0, 10050.0, 50.0]
    pruned_doc = upsert_ohlc_bar_archive(
        cache_dir,
        "SOLUSDC",
        [old_bar, *btc_1h_bars[-5:]],
        timeframe="1h",
        source="binance:SOLUSDC",
        now_ts=t_inc,
        max_retention_days=365,
    )
    assert pruned_doc is not None and pruned_doc["bar_count"] == 5

    # 7. chart_view CLI supports --tf 4h and --tf 1h directly from persisted archives
    assert chart_view_main(["BTCUSDC", "--tf", "4h", "--months", "1", "--cache-dir", cache_dir]) == 0
    out_4h = capsys.readouterr().out
    assert "=== BTCUSDC (4h," in out_4h
    assert chart_view_main(["NVDA_US_EQ", "--tf", "1h", "--months", "1", "--cache-dir", cache_dir]) == 0
    out_1h = capsys.readouterr().out
    assert "=== NVDA_US_EQ (1h," in out_1h

    # 8. Review fixes: cross-venue isolation, weekend updated_ts refresh, Yahoo 1d closed bar, min_bars floor, intraday ZigZag
    from providers.ohlc_bar_archive import load_ohlc_archive_dataset

    # 8a. Cross-venue isolation: Kraken request must NEVER receive Hyperliquid/Binance archive bars
    assert (
        load_ohlc_archive_dataset(
            cache_dir,
            "BTCUSDC",
            "4h",
            lookback_days=5,
            now_ts=t_inc,
            expected_source_prefix="kraken",
        )
        is None
    )
    assert (
        load_ohlc_archive_dataset(
            cache_dir,
            "BTCUSDC",
            "4h",
            lookback_days=5,
            now_ts=t_inc,
            expected_source_prefix="binance",
        )
        is not None
    )

    # 8b. Weekend/flat-market updated_ts refresh: unchanged bars after >=30m still refresh updated_ts on disk
    fri_ts = datetime(2026, 11, 20, 22, 0, 0, tzinfo=timezone.utc).timestamp()
    sat_ts = fri_ts + 4 * 3600.0
    eq_1d = [[int(datetime(2026, 11, 20, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000), 150.0, 152.0, 149.0, 151.0, 1000.0]]
    d_fri = upsert_ohlc_bar_archive(cache_dir, "SPY", eq_1d, timeframe="1d", source="yahoo:SPY", now_ts=fri_ts)
    d_sat = upsert_ohlc_bar_archive(cache_dir, "SPY", eq_1d, timeframe="1d", source="yahoo:SPY", now_ts=sat_ts)
    assert d_fri is not None and d_sat is not None and d_sat["updated_ts"] == round(sat_ts, 3)

    # 8c. Yesterday's closed Yahoo 1d bar (14:30 UTC) is NOT falsely dropped at 10:00 UTC next morning
    next_morning_ms = int(datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert len(filter_closed_archive_bars(eq_1d, interval_min=1440, now_ms=next_morning_ms)) == 1

    # 8d. Partial tiny archive (<20 bars) returns None when lookback_days is omitted (walk-forward floor)
    assert load_ohlc_archive_dataset(cache_dir, "SOLUSDC", "1h", lookback_days=None, now_ts=t_inc) is None

    # 8e. Intraday ZigZag preserves sub-daily pivots (not collapsed to 1 bar/day) with MM-DD labels
    intra_base = int(datetime(2026, 11, 25, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    intra_wave = [
        [intra_base + 0 * 3600 * 1000, 100.0, 100.0, 100.0, 100.0, 10.0],
        [intra_base + 4 * 3600 * 1000, 105.0, 112.0, 105.0, 112.0, 10.0],
        [intra_base + 8 * 3600 * 1000, 106.0, 106.0, 98.0, 98.0, 10.0],
        [intra_base + 12 * 3600 * 1000, 104.0, 110.0, 104.0, 110.0, 10.0],
    ]
    sw_intra = compute_zigzag_swings(intra_wave, threshold_pct=5.0, max_pivots=8, preserve_intraday=True)
    assert len(sw_intra) >= 3 and sw_intra[0]["ym"] == "11-25"
