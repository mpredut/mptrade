"""Unit and integration tests for configurable multi-month/annual structure, operator strategic notes, and chart_view zoom."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from botcore import topic_spec
from cachepaths import CACHE_LONG_HORIZON_FILENAME, OPERATOR_REGIME_NOTE_FILENAME, resolve_cachedb_filepath
from intelligence.macro_sentiment_ssot import TelemetrySensorSnapshot
from intelligence.sentiment.collectors.long_horizon_collector import (
    collect_long_horizon_snapshot,
    load_active_operator_regime_note,
)
from intelligence.sentiment.evaluators.long_horizon_evaluator import (
    compute_zigzag_swings,
    evaluate_symbol_long_horizon,
    format_horizon_label,
    format_long_horizon_section_lines,
    parse_operator_regime_note_text,
    recalculate_long_horizon_with_live_price,
    render_ascii_chart_view,
)
from intelligence.sentiment.evaluators.prompt_delta_evaluator import evaluate_prompt_delta_plan
from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
from orchestratorOS.supervisor.crypto_executor_delta_gate import evaluate_crypto_executor_delta_gate
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt
from verify_tools.audit_prompt_invariants import audit_rendered_prompt_text
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

    # ASCII chart zoom supports both --months and --from/--to slices cleanly
    chart_txt = render_ascii_chart_view(ev_18m, start_ym="2026-01", end_ym="2026-10", timeframe="1w")
    assert "=== BTCUSDC (1w," in chart_txt
    assert "Swings:" in chart_txt
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
    assert "Multi-Month & Annual Structure (18m / 18m lookback" in cold
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

    # Verify CLI chart_view works end-to-end on persisted snapshot
    rc = chart_view_main(["BTCUSDC", "--months", "12", "--cache-dir", cache_dir])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "=== BTCUSDC (1w," in captured
