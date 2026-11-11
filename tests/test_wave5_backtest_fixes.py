"""Targeted regression tests for Wave 5 backtest, replay, runner, and research fixes."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

WAVE5_FILES = (
    # Batch A
    "strategies/position_exit_manager.py",
    "strategies/spot_replay.py",
    "hyperliquid/replay.py",
    "hyperliquid/backtest.py",
    "kraken/backtest_trail.py",
    "212trading/analyze_volatility.py",
    # Batch B
    "offline/backtests/comparison.py",
    "offline/backtests/execution_calibration.py",
    "offline/backtests/financial_benchmark.py",
    "offline/backtests/promotion.py",
    "offline/backtests/hype_candidates.py",
    "offline/backtests/metrics.py",
    "offline/backtests/walk_forward.py",
    "offline/backtests/tradeall.py",
    "offline/research/hype_dataset/financial_baseline_v1.json",
    # Batch C
    "offline/runners/t212_walk_forward_baseline.py",
    "offline/runners/walk_forward_compare.py",
    "offline/runners/binance_walk_forward_baseline.py",
    "offline/runners/hyperliquid_walk_forward_baseline.py",
    "offline/runners/kraken_financial_benchmark.py",
    "offline/runners/kraken_financial_compare.py",
    "offline/runners/spot_profile_review.py",
    "offline/runners/fetch_hyperliquid_candles.py",
    "offline/runners/apply_proposals.py",
    "offline/research/monitortrades_backtest/scheduled_pilot.py",
    "offline/runners/run_backtest_cycle.sh",
    "offline/runners/run_backtest_parallel.sh",
    "offline/runners/run_financial_baseline.sh",
    # Batch D
    "offline/research/intelligence_backtest.py",
    "offline/research/chop_weight_backtest.py",
    "offline/research/verify_optimizations_parity.py",
    "offline/research/compare_mhdsl_reentry_replay.py",
    "offline/research/backtest_ranges.py",
    "offline/research/short_trend_estimator_benchmark.py",
    "offline/research/hybrid_reentry/run_suite.py",
    "offline/research/kraken_continuous_grid/fetch_data.py",
    "offline/research/kraken_continuous_grid/grid.py",
    "offline/research/kraken_continuous_grid/grid2.py",
    "offline/research/kraken_continuous_grid/grid3.py",
    "offline/research/kraken_continuous_grid/report3.py",
    "offline/research/t212_continuous_grid/fetch.py",
    "offline/simulations/sim.py",
    "offline/simulations/sim2.py",
    "tests/test_wave5_backtest_fixes.py",
)


def test_wave5_ascii_and_shell_syntax_hygiene():
    for rel in WAVE5_FILES:
        path = ROOT / rel
        assert path.is_file(), f"Missing file: {rel}"
        text = path.read_text(encoding="utf-8")
        bad = [(idx + 1, ch) for idx, line in enumerate(text.splitlines()) for ch in line if ord(ch) >= 128]
        assert not bad, f"Non-ASCII characters in {rel}: {bad[:5]}"
        if rel.endswith(".sh"):
            proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
            assert proc.returncode == 0, f"bash -n failed for {rel}: {proc.stderr}"


def test_batch_a_position_exit_manager_zero_pullback_trend_bypass():
    from strategies.position_exit_manager import (
        PositionExitConfig,
        PositionExitManager,
        ReentrySnapshot,
    )

    snap = ReentrySnapshot(
        symbol="BTCUSD",
        current_price=105.0,
        last_sell_price=100.0,
        last_exit_kind="TP",
        post_sell_peak=104.0,
        post_sell_trough=99.0,
        current_time_s=2000.0,
        last_sell_time_s=1000.0,
        trend_up=True,
    )

    # Zero pullback ("Trend Bypass") confirms immediately in a bull trend
    cfg_zero = PositionExitConfig(
        reentry_hybrid_enabled=True,
        reentry_peak_relative=True,
        reentry_pullback_pct=0.0,
        reentry_ratchet_enabled=False,
        reentry_drop_pct=0.02,
    )
    res_zero = PositionExitManager(cfg_zero).evaluate_reentry(snap)
    assert res_zero.should_enter is True
    assert "bull_reentry_pullback_confirmed" in res_zero.reason

    # Non-zero pullback without actual pullback from peak stays unconfirmed
    cfg_wait = PositionExitConfig(
        reentry_hybrid_enabled=True,
        reentry_peak_relative=True,
        reentry_pullback_pct=0.02,
        reentry_ratchet_enabled=False,
        reentry_drop_pct=0.02,
    )
    res_wait = PositionExitManager(cfg_wait).evaluate_reentry(snap)
    assert res_wait.should_enter is False

    # Exit trailing helper preserves strict > 0 threshold contract
    assert snap.has_pulled_back(105.0, 0.0) is False


def test_batch_a_spot_replay_cycle_wins_and_finite_cash():
    from strategies.spot_engine import StratParams
    from strategies.spot_replay import run_replay

    # Multi-tranche TP cycle counts at most 1 win per closed cycle (wins <= cycles)
    params = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=50.0,
        dca_drop_pct=5.0,
        check_minutes=60.0,
        takeprofit_pct=10.0,
        max_budget=300.0,
        max_dca_buys=2,
        enable_takeprofit=True,
        order_ttl_min=120.0,
        tp_tranches=[(3.0, 50.0), (6.0, 50.0)],
        reentry_drop_pct=50.0,
    )
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 104.0, 99.8, 103.5),
        (103.5, 108.0, 103.0, 107.0),
        (107.0, 112.0, 106.5, 111.0),
    ]
    rep = run_replay(ohlc, params, fee_pct=0.1, bar_minutes=60, initial_cash=300.0)
    assert rep["cycles"] == 1
    assert rep["wins"] == 1
    assert rep["wins"] <= rep["cycles"]


def test_batch_a_hyperliquid_replay_multihour_funding_and_sl_vs_dca():
    from offline.backtests.evaluation import load_venue_module

    hl_rp = load_venue_module("hyperliquid", "replay.py", "hl_replay_wave5_test")

    # 1. 4h bar sums all 4 hourly funding sub-steps (including timestamps >= 10_000_000_000 ms)
    t0 = 11_000_000.0  # seconds whose ms representation exceeds 10_000_000_000
    ts = [t0, t0 + 14400.0, t0 + 28800.0, t0 + 43200.0]
    ohlc = [
        (100.0, 100.2, 99.8, 100.0),  # bar 0 places entry limit at 100
        (100.0, 100.2, 99.8, 100.0),  # bar 1 fills entry at 100
        (100.0, 100.2, 99.8, 100.0),  # bar 2 accrues 4h of funding
        (100.0, 100.2, 99.8, 100.0),  # bar 3 accrues 4h of funding
    ]
    sig = ["LONG", "LONG", "LONG", "LONG"]
    funding = [(int((t0 + 3600.0 * k) * 1000), 0.001) for k in range(16)]
    p_fund = hl_rp.StratParams(
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=50.0,
        dca_drop_pct=10.0,
        max_dca_buys=0,
        max_budget=100.0,
        takeprofit_pct=50.0,
        stop_loss_pct=50.0,
        enable_takeprofit=True,
        fee_pct=0.0,
    )
    res_f = hl_rp.run_replay(ohlc, ts=ts, fund=funding, sig=sig, params=p_fund, direction="long", bar_minutes=240.0)
    assert res_f["funding"] == pytest.approx(-0.8, rel=1e-3)

    # 2. Resting DCA limit below stop-loss level does not fill after SL triggers on crash bar
    p_sl = hl_rp.StratParams(
        entry_amount=100.0,
        entry_discount_pct=0.0,
        dca_amount=100.0,
        dca_drop_pct=10.0,
        max_dca_buys=2,
        max_budget=300.0,
        takeprofit_pct=20.0,
        stop_loss_pct=5.0,
        enable_takeprofit=True,
        fee_pct=0.0,
    )
    ohlc_crash = [
        (100.0, 100.5, 99.5, 100.0),  # bar 0 places entry at 100
        (100.0, 100.2, 99.8, 100.0),  # bar 1 fills entry at 100 and places resting DCA at 90 (SL is 95)
        (100.0, 100.0, 85.0, 86.0),   # bar 2 crashes through SL=95 down to 85 -> SL must exit before DCA=90
    ]
    ts_crash = [1000.0, 1060.0, 1120.0]
    res_sl = hl_rp.run_replay(
        ohlc_crash, ts=ts_crash, sig=["LONG", "LONG", "LONG"], params=p_sl, direction="long", bar_minutes=1.0
    )
    assert res_sl["cycles"] == 1
    assert res_sl["fills"] == 2  # 1 entry fill + 1 SL exit fill; DCA at 90 was skipped/cancelled
    assert res_sl["realized"] == pytest.approx(-5.0, abs=1e-4)


def test_batch_b_comparison_pareto_all_invalid_and_bool_window_exclusion():
    from offline.backtests.comparison import pareto_frontier, relative_to_baseline

    bad_items = {
        "nan_cand": {"mean_return_pct": None, "worst_max_drawdown_pct": 5.0},
        "neg_dd": {"mean_return_pct": 3.0, "worst_max_drawdown_pct": -1.0},
    }
    assert pareto_frontier(bad_items) == []

    base_wins = [
        {"window": 1, "return_pct": 2.0},
        {"window": True, "return_pct": 999.0},
    ]
    cand_wins = [
        {"window": 1, "return_pct": 4.0},
        {"window": 1, "return_pct": 100.0},   # duplicate window ignored
        {"window": True, "return_pct": -999.0},  # boolean window ignored
    ]
    rel = relative_to_baseline(cand_wins, base_wins, key_field="window")
    assert rel["wins_vs_live"] == 1
    assert rel["ties_vs_live"] == 0
    assert rel["losses_vs_live"] == 0
    assert rel["mean_return_delta_pct"] == pytest.approx(2.0)


def test_batch_b_execution_calibration_partial_cancel_cost_and_market_exclusion():
    from offline.backtests.execution_calibration import calibrate_execution_events

    valid_cid = "a" * 32
    events = [
        # Limit order partially filled then terminal-canceled with cost=0, fee=0
        {
            "event": "submit_requested",
            "intent_id": "INT-LIM",
            "venue": "Kraken",
            "symbol": "BTCUSD",
            "side": "BUY",
            "market": False,
            "price": 100.0,
            "qty": 2.0,
            "ts": 1000.0,
        },
        {
            "event": "submit_accepted",
            "intent_id": "INT-LIM",
            "venue": "Kraken",
            "symbol": "BTCUSD",
            "order_id": "ORD-1",
            "client_order_id": valid_cid,
            "ts": 1001.0,
        },
        {
            "event": "order_status",
            "intent_id": "INT-LIM",
            "order_id": "ORD-1",
            "filled_qty": 1.0,
            "cost": 99.8,
            "fee": 0.25,
            "ts": 1005.0,
        },
        {
            "event": "order_terminal",
            "intent_id": "INT-LIM",
            "order_id": "ORD-1",
            "status": "canceled",
            "filled_qty": 1.0,
            "cost": 0.0,
            "fee": 0.0,
            "ts": 1010.0,
        },
        # Market order with non-null price must NOT enter limit_fill_deviation_bps
        {
            "event": "submit_requested",
            "intent_id": "INT-MKT",
            "venue": "kraken",
            "symbol": "BTCUSD",
            "side": "SELL",
            "market": True,
            "price": 102.0,
            "reference_price": 102.0,
            "qty": 1.0,
            "ts": 900.0,
        },
        {
            "event": "submit_accepted",
            "intent_id": "INT-MKT",
            "venue": "kraken",
            "symbol": "BTCUSD",
            "order_id": "ORD-2",
            "client_order_id": valid_cid,
            "ts": 901.0,
        },
        {
            "event": "order_terminal",
            "intent_id": "INT-MKT",
            "order_id": "ORD-2",
            "filled_qty": 1.0,
            "cost": 101.8,
            "fee": 0.26,
            "ts": 902.0,
        },
    ]
    cal = calibrate_execution_events(events)
    by_intent = {o["intent_id"]: o for o in cal["orders"]}
    assert by_intent["INT-LIM"]["fee_bps"] == pytest.approx(0.25 / 99.8 * 10_000)
    assert by_intent["INT-LIM"]["limit_fill_deviation_bps"] == pytest.approx((99.8 / 100.0 - 1.0) * 10_000)

    # Market order excluded from limit_fill_deviation_bps in 'all' summary
    assert cal["summary"]["all"]["limit_fill_deviation_bps"]["count"] == 1
    assert cal["summary"]["market"]["limit_fill_deviation_bps"]["count"] == 0

    # Case-insensitive venue deduplication picks earliest accepted_ts (901.0)
    fv = cal["client_order_id_validation"]["first_valid_by_venue"]
    assert len(fv) == 1
    only_entry = next(iter(fv.values()))
    assert only_entry["intent_id"] == "INT-MKT"
    assert only_entry["accepted_ts"] == pytest.approx(901.0)


def test_batch_b_metrics_absorbing_ruin_and_promotion_zero_base_and_assumptions():
    from offline.backtests.metrics import calculate_performance_metrics, run_tp_drop_sl_sweep
    from offline.backtests.promotion import _assumption_signature, _relative_change

    # Absorbing ruin: equity hits 0 at bar 1, goes negative at bar 2, stays <= 0 at bar 3
    perf = calculate_performance_metrics(
        [100.0, 0.0, -50.0, -25.0],
        initial_capital=100.0,
        periods_per_year=365.0,
    )
    assert perf["annualized_return_pct"] == pytest.approx(-100.0)
    assert perf["sharpe"] is not None and perf["sharpe"] < 0.0

    with pytest.raises(ValueError, match="budget must be positive"):
        run_tp_drop_sl_sweep({}, lambda _p: {}, budget=0.0)

    assert _relative_change(0.0, 5.0) is None
    sig_ro = _assumption_signature({"maker_fee_pct": 0.16, "taker_fee_pct": 0.26, "description": "text vechi"})
    sig_en = _assumption_signature({"maker_fee_pct": 0.16, "taker_fee_pct": 0.26, "description": "new text"})
    assert sig_ro == sig_en


def test_batch_b_tradeall_backtest_positive_order_ids_and_state_isolation(tmp_path):
    from offline.backtests import tradeall as bt_ta

    clock = bt_ta.SimClock()
    clock.ts = 1700000000.0
    broker = bt_ta.BacktestBroker(str(tmp_path), clock)
    broker.current_tick_price = 100.0
    o1 = broker.place_order_smart("BUY", "BTCUSDC", 101.0, 1.0)
    o2 = broker.place_order_smart("SELL", "BTCUSDC", 99.0, 0.5)
    assert o1 is not None and o1["orderId"] == 1
    assert o2 is not None and o2["orderId"] == 2
    assert broker.last_price == pytest.approx(100.0)

    orig_fire = set(bt_ta.ta.TRADEALL_FIRE_SYMBOLS)
    jsonl = tmp_path / "ticks.jsonl"
    jsonl.write_text(
        '{"s": "TESTPAIRUSDC", "i": [1700000000000, 100.0]}\n'
        '{"s": "TESTPAIRUSDC", "i": [1700000005000, 100.5]}\n',
        encoding="utf-8",
    )
    run_id = "wave5_pytest_tmp_isolation"
    try:
        bt_ta.run_backtest(
            "TESTPAIRUSDC",
            1700000000.0,
            1700000010.0,
            "fast",
            run_id,
            "cache24",
            cache24_file=str(jsonl),
            quiet=True,
        )
        assert bt_ta.ta.TRADEALL_FIRE_SYMBOLS == orig_fire
    finally:
        shutil.rmtree(ROOT / "logger" / "backtest" / run_id, ignore_errors=True)


def test_batch_c_classic_tp_disables_mdhtp_and_pilot_missing_key_guard(tmp_path, monkeypatch):
    from offline.backtests.hype_candidates import Candidate, default_candidates
    from offline.research.monitortrades_backtest import scheduled_pilot as sp

    raw_ov = {"takeprofit_pct": 4.0}
    cand = Candidate("custom", "desc", raw_ov)
    raw_ov["takeprofit_pct"] = 99.0
    assert cand.overrides["takeprofit_pct"] == 4.0

    cands = {c.name: c.overrides for c in default_candidates()}
    assert cands["classic_tp"] == {
        "tp_trend_hold": False,
        "tp_mdhtp_enabled": False,
        "tp_dynamic_flat": False,
    }

    missing_conf = tmp_path / "nonexistent.conf"
    monkeypatch.setattr(sp, "INSTRUMENTS_CONF", str(missing_conf))
    with pytest.raises(FileNotFoundError):
        sp._apply_config_change("BINANCE_BTC", "mt.gain", 7.0, 8.0)
    assert not missing_conf.exists()

    existing_conf = tmp_path / "instruments.conf"
    existing_conf.write_text("[BINANCE_BTC]\nmt.gain = 7.0\n", encoding="utf-8")
    monkeypatch.setattr(sp, "INSTRUMENTS_CONF", str(existing_conf))
    with pytest.raises(KeyError):
        sp._apply_config_change("BINANCE_BTC", "mt.unknown", 7.0, 8.0)
    assert existing_conf.read_text(encoding="utf-8") == "[BINANCE_BTC]\nmt.gain = 7.0\n"
    sp._apply_config_change("BINANCE_BTC", "mt.gain", 7.0, 8.0)
    assert "mt.gain = 8.0" in existing_conf.read_text(encoding="utf-8")


def test_batch_d_chop_intelligence_short_trend_report3_and_sim_window(tmp_path, monkeypatch):
    from offline.research import chop_weight_backtest as cwb
    from offline.research import intelligence_backtest as ib
    from offline.research import short_trend_estimator_benchmark as stb
    from offline.research.kraken_continuous_grid import report3
    from offline.simulations.sim import PriceWindow
    from state_io import atomic_write_jsonl

    # 1. Chop backtest rejects invalid policy even on empty series and guards NaN weights
    with pytest.raises(ValueError, match="Unknown chop policy"):
        cwb.run_single_simulation("TAOUSDC", [], None, "TYPO_POLICY")

    sampled = [(1700000000.0 + i * 300.0, 100.0 + math.sin(i / 4.0) * 5.0 + i * 0.2) for i in range(120)]
    btc_15m = [(1700000000.0 + i * 900.0, 50000.0 + i * 150.0) for i in range(45)]
    with patch.object(cwb, "get_trade_weight", return_value=([0.0], [float("nan")])):
        st_nan = cwb.run_single_simulation("TAOUSDC", sampled, btc_15m, "PROXY_BTC", step_seconds=300)
        assert math.isfinite(st_nan.final_balance)

    # 2. Intelligence backtest excludes current bar from local_high_48 and computes causal survival
    bars_55 = [(1700000000.0 + i * 300.0, 100.0 + (20.0 if i == 54 else 0.0)) for i in range(55)]
    tape = ib._precompute_intelligence_tape("BTCUSDC", bars_55, step_seconds=300, parabolic_surge_pct=3.5)
    assert tape[-1]["p"] == pytest.approx(120.0)
    assert tape[-1]["local_high_48"] == pytest.approx(100.0)

    synth_tape = []
    for seg in range(6):
        dur_days = seg + 1
        synth_tape.append({"current_trend": "BULL", "trend_duration_sec": dur_days * 86400.0})
        synth_tape.append({"current_trend": "FLAT", "trend_duration_sec": 0.0})
    sched = ib._compute_causal_survival_schedule(synth_tape, warmup_median_days=3.0, warmup_p90_days=7.0)
    assert sched[0] == (3.0, 7.0)  # Warmup prior before 5 completed trends
    assert sched[-1][0] == pytest.approx(3.5)  # Median of completed durations [1, 2, 3, 4, 5, 6]

    # 3. Short-trend benchmark IS metrics are invariant to OOS price mutations
    n_pts = 600
    ts = np.arange(n_pts, dtype=np.float64)
    rng = np.random.default_rng(7)
    px_a = 1000.0 + np.cumsum(rng.normal(0.0, 1.5, size=n_pts))
    px_b = px_a.copy()
    t_split = ts[0] + stb.IS_FRAC * (ts[-1] - ts[0])
    i_split = int(np.searchsorted(ts, t_split))
    px_b[i_split:] = px_b[i_split:] * 3.5 + 500.0

    monkeypatch.setattr(stb, "load_series", lambda _sym: (ts, px_a))
    _, _, res_a = stb.evaluate("SYNTH")
    monkeypatch.setattr(stb, "load_series", lambda _sym: (ts, px_b))
    _, _, res_b = stb.evaluate("SYNTH")
    for k in ("sp60", "sp180", "hit", "flips", "lag_med", "buy", "sell", "ride"):
        va = res_a["LEGACY (sf+gr)/2"]["IS"][k]
        vb = res_b["LEGACY (sf+gr)/2"]["IS"][k]
        if math.isnan(va):
            assert math.isnan(vb)
        else:
            assert va == pytest.approx(vb, rel=1e-12, abs=1e-12), f"IS leakage in {k}: {va} vs {vb}"

    # 4. report3 renders '-/-' when 'live' row is absent instead of counting losses as beating -inf
    exp_dir = tmp_path / "exp3"
    exp_dir.mkdir()
    atomic_write_jsonl(
        str(exp_dir / "results3.jsonl"),
        [{"asset": "BTC", "cfg": "sl0", "0.26": {"ret": -12.5, "dd": 20.0}, "0.4": {"ret": -15.0, "dd": 22.0}}],
    )
    lines = report3.generate_report(str(exp_dir))
    assert any("| sl0 | -/- |" in line for line in lines)

    # 5. sim.PriceWindow rejects non-integer window_size and keeps duplicate lows across window expiry
    with pytest.raises(ValueError):
        PriceWindow(4.7)
    pw = PriceWindow(window_size=3, epsilon=1e-5)
    for px in (10.0, 20.0, 10.0 + 5e-6):
        pw.process_price(px)
    assert pw.get_min()[0] == pytest.approx(10.0)
    pw.process_price(30.0)  # Index 0 (10.0) expires; index 2 (10.000005) is still in window!
    assert pw.get_min()[0] == pytest.approx(10.0 + 5e-6)
