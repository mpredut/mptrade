#!/usr/bin/env python3
"""tests/test_research_and_backtest_robustness.py — Regression tests for research runners, backtesting, and reporting robustness."""

from __future__ import annotations

import json
import os
import pytest

from offline.backtests import execution_calibration, financial_benchmark
from offline.research.chop_weight_backtest import run_single_simulation
from offline.research.hybrid_reentry import harness as hybrid_harness
from offline.research.hybrid_reentry import report as hybrid_report
from offline.research.t212_continuous_grid import report as t212_report
from strategies import spot_engine


@pytest.fixture(autouse=True)
def restore_strategy_step():
    orig = spot_engine.Strategy.step
    yield
    spot_engine.Strategy.step = orig


def test_t212_report_empty_directory(tmp_path):
    """Calling generate_report on an empty directory returns clean markdown without crashing."""
    out = t212_report.generate_report(str(tmp_path))
    assert "# Trading 212 grid" in out
    assert "No results found" in out


def test_t212_report_with_missing_live_baseline_and_none_metrics(tmp_path):
    """Reporting handles data where live baseline is missing or contains None metrics."""
    results_file = tmp_path / "results.jsonl"
    data = [
        {"profile": "nvda", "variant": "tp5", "asset": "NVDA", "1h": {"ret": 10.5, "dd": 2.1}, "w90": [[1.2, 0.5, 3.0]]},
        {"profile": "nvda", "variant": "tp5", "asset": "SPY", "1h": {"ret": 5.0, "dd": 1.2}, "1d": {"ret": None, "dd": None}},
        {"profile": "spcx", "variant": "tp8", "asset": "SPCX", "1h": None, "1d": None},
        {"profile": "rgnt", "variant": "live", "asset": "RGNT", "error": "test error trace\nValueError: test"},
    ]
    with open(results_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    out = t212_report.generate_report(str(tmp_path))
    assert "# Trading 212 grid" in out
    assert "Profile `nvda`" in out
    assert os.path.exists(tmp_path / "REPORT.md")


def test_hybrid_reentry_report_none_metrics(tmp_path, monkeypatch):
    """Hybrid reentry report handles None metric entries without format string TypeError."""
    crypto_json = tmp_path / "ablation_results.json"
    data = [
        {
            "asset": "BTC",
            "test_id": "Test 0",
            "variant": "Baseline",
            "total_return_pct": None,
            "max_drawdown_pct": None,
            "exposure_pct": None,
        }
    ]
    with open(crypto_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    monkeypatch.setattr(hybrid_report, "CRYPTO_JSON", str(crypto_json))
    monkeypatch.setattr(hybrid_report, "SUMMARY_JSON", str(tmp_path / "missing_summary.json"))
    monkeypatch.setattr(hybrid_report, "EQUITIES_JSON", str(tmp_path / "missing_equities.json"))
    rep = hybrid_report.generate_markdown_report()
    assert "Hybrid Re-Entry Architecture" in rep
    assert "**BTC**" in rep
    assert "N/A" in rep


def test_hybrid_harness_empty_ohlc():
    """Hybrid harness returns empty list cleanly when given empty OHLC."""
    from offline.research.hybrid_reentry.run_suite import BASE_PARAMS
    res = hybrid_harness.evaluate_asset("BTC", [], BASE_PARAMS)
    assert res == []


def test_chop_weight_backtest_empty_sampled():
    """Chop weight simulation handles empty sampled list without crashing or IndexError."""
    stats = run_single_simulation("TEST", [], None, "CHOP_0.03")
    assert stats.symbol == "TEST"
    assert stats.total_trades == 0
    assert stats.net_profit_usd == 0.0


def test_financial_benchmark_none_net_pnl():
    """aggregate_financial_windows handles windows whose metrics have None net_pnl."""
    windows = [
        {
            "key": "test_win",
            "return_pct": 2.5,
            "max_drawdown_pct": 1.0,
            "buy_hold_return_pct": 1.5,
            "cycles": 1,
            "fills": 2,
            "metrics": {
                "net_pnl": None,
                "max_drawdown_abs": 10.0,
            },
        }
    ]
    agg = financial_benchmark.aggregate_financial_windows(windows, initial_capital=1000.0)
    assert agg["sum_reset_net_pnl_usd"] == 0.0
    assert agg["mean_net_pnl_usd_per_window"] == 0.0


def test_execution_calibration_distribution_none_handling():
    """distribution() handles None values in list without TypeError."""
    dist = execution_calibration.distribution([1.0, None, 2.0, None, 3.0])
    assert dist["count"] == 3
    assert dist["min"] == 1.0
    assert dist["max"] == 3.0
    assert dist["mean"] == 2.0


def test_scheduled_pilot_run_one_delegates_to_run_symbol():
    """scheduled_pilot._run_one delegates directly to run_replay_backtest.run_symbol(..., series=...)."""
    from offline.research.monitortrades_backtest import run_replay_backtest as rb
    from offline.research.monitortrades_backtest import scheduled_pilot as sp

    cfg = rb.SYMBOLS["BTCUSDC"]
    series = [(1_700_000_000.0 + i * 60.0, 50_000.0 + (i % 5) * 100.0) for i in range(30)]
    full_res = rb.run_symbol("BTCUSDC", cfg["params"], cfg["base"], quiet=True, series=series)
    pilot_res = sp._run_one("BTCUSDC", cfg["base"], cfg["params"], series)
    assert full_res is not None
    assert pilot_res == {"net": full_res["net"], "buy_hold": full_res["buy_hold"]}
    assert sp._run_one("BTCUSDC", cfg["base"], cfg["params"], []) is None


def test_spot_replay_adapters_ssot():
    """All spot replay adapters delegate to strategies.spot_replay.make_spot_replay_adapter."""
    import importlib.util
    from pathlib import Path
    from binance_api.replay import make_binance_replay_adapter
    from kraken.replay import make_kraken_replay_adapter
    from strategies.spot_engine import StratParams
    from strategies.spot_replay import make_spot_replay_adapter

    hl_path = Path(__file__).resolve().parents[1] / "hyperliquid" / "replay.py"
    spec = importlib.util.spec_from_file_location("hl_replay_ssot_test", str(hl_path))
    hl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl_mod)
    make_hyperliquid_spot_replay_adapter = hl_mod.make_hyperliquid_spot_replay_adapter

    params = StratParams(
        currency="USDC",
        entry_amount=100.0,
        entry_discount_pct=0.5,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        takeprofit_pct=2.0,
        max_budget=300.0,
        max_dca_buys=2,
        stop_loss_pct=0.0,
        check_minutes=1.0,
        enable_takeprofit=True,
        order_ttl_min=60.0,
    )
    ohlc = [(100.0, 101.0, 99.0, 100.0), (100.0, 102.0, 98.0, 101.0)]
    for factory in (
        make_spot_replay_adapter,
        make_kraken_replay_adapter,
        make_binance_replay_adapter,
        make_hyperliquid_spot_replay_adapter,
    ):
        adapter = factory(params, bar_minutes=60.0)
        res = adapter(ohlc, None)
        assert "return_pct" in res
        assert "max_drawdown_pct" in res


def test_sim_clock_value_and_ts_alias():
    """SimClock supports both .ts and .value property access and optional initial_ts."""
    from providers.replay_clock import SimClock

    clock = SimClock(0.0)
    assert clock.ts == 0.0
    assert clock.value == 0.0
    assert clock() == 0.0
    clock.value = 123.5
    assert clock.ts == 123.5
    assert clock() == 123.5


def test_walk_forward_cli_shared_helpers(tmp_path):
    """Shared walk-forward CLI helpers parse intervals, dataset specs, and window sizes."""
    import argparse
    from offline.backtests.evaluation import (
        add_walk_forward_cli_args,
        build_execution_model_from_args,
        parse_dataset_specs,
        parse_intervals,
        resolve_window_sizes,
    )

    assert parse_intervals("60, 240,1440") == [60, 240, 1440]
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
    specs = parse_dataset_specs([f"60={csv_file}"])
    assert specs[60] == csv_file.resolve()

    parser = argparse.ArgumentParser()
    add_walk_forward_cli_args(parser, include_warmup=True, default_warmup=0)
    args = parser.parse_args(["--train", "50", "--validation", "10", "--test", "15", "--spread-bps", "5"])
    model = build_execution_model_from_args(parser, args)
    assert model.spread_bps == 5.0
    assert resolve_window_sizes(args, 200) == (50, 10, 15, 15)


def test_strat_params_from_dict_ignores_removed_fields():
    """StratParams.from_dict merges defaults and drops removed legacy fields."""
    from strategies.spot_engine import StratParams

    defaults = {
        "currency": "USDC",
        "entry_amount": 100.0,
        "entry_discount_pct": 0.2,
        "dca_amount": 50.0,
        "dca_drop_pct": 2.0,
        "check_minutes": 1.0,
        "takeprofit_pct": 3.0,
        "max_budget": 500.0,
        "max_dca_buys": 5,
        "enable_takeprofit": True,
        "order_ttl_min": 60.0,
    }
    p = StratParams.from_dict(
        {"takeprofit_pct": 4.5, "removed_legacy_field": 123},
        defaults=defaults,
        warn_dropped=True,
    )
    assert p.takeprofit_pct == 4.5
    assert p.currency == "USDC"


def test_t212_replay_adapter_historical_fx_alignment():
    """make_t212_replay_adapter aligns fx_records via offline.backtests.datasets without ModuleNotFoundError."""
    import importlib.util
    import sys
    from offline.backtests.evaluation import evaluate_segment

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "t212_replay_fx_test", os.path.join(root_dir, "212trading", "replay.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    params = mod._strat.StratParams(
        currency="RON",
        entry_amount=100.0,
        entry_discount_pct=0.2,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        check_minutes=5.0,
        takeprofit_pct=2.0,
        max_budget=1000.0,
        max_dca_buys=2,
        validity="GOOD_TILL_CANCEL",
        enable_takeprofit=True,
        order_ttl_min=10.0,
        stop_loss_pct=5.0,
        yahoo_sym="TEST",
        reentry_drop_pct=0.0,
        reentry_tolerance_pct=0.05,
        tp_ladder=[],
        fx_fee_pct=0.15,
        loss_alert_step=1.0,
        ladder_min_free=6.0,
        sl_rebuy_enabled=False,
        sl_rebuy_bounce_pct=1.2,
    )
    records = [
        {"timestamp": 1700000000 + i * 86400, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}
        for i in range(5)
    ]
    fx_records = [{"timestamp": 1699900000, "close": 4.5}]
    adapter = mod.make_t212_replay_adapter(
        params, bar_minutes=1440.0, fx_records=fx_records, fx_invert=True
    )
    seg = evaluate_segment(records, adapter, validate_contract=True)
    assert seg["metrics"]["account_currency"] == "RON"


def test_hyperliquid_funding_at_seconds_to_ms_and_param_immutability():
    """funding_at normalises Unix seconds to milliseconds and run_replay does not mutate StratParams."""
    import importlib.util

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "hl_replay_unit_test", os.path.join(root_dir, "hyperliquid", "replay.py")
    )
    hl_replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl_replay)

    fund = [(1700000000000, 0.0001), (1700003600000, 0.0005)]
    # Passing timestamp in seconds (1700003600) must match the second funding entry (0.0005)
    assert hl_replay.funding_at(fund, 1700003600) == 0.0005
    assert hl_replay.funding_at(fund, 1700003600000) == 0.0005

    orig_params = hl_replay.StratParams(fee_pct=0.035)
    ohlc = [(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)]
    hl_replay.run_replay(ohlc, params=orig_params, fee_pct=0.08)
    assert orig_params.fee_pct == 0.035


def test_tradeall_iter_jsonl_ticks_does_not_abort_on_out_of_order_end_ts(tmp_path):
    """_iter_jsonl_ticks skips out-of-order lines with ts > end_ts instead of terminating early."""
    import json
    from offline.backtests.tradeall import _iter_jsonl_ticks

    p = tmp_path / "ticks.jsonl"
    lines = [
        {"s": "BTCUSDC", "i": [1000_000, 100.0]},
        {"s": "BTCUSDC", "i": [5000_000, 999.0]},  # out-of-order future spike > end_ts
        {"s": "BTCUSDC", "i": [2000_000, 101.0]},  # valid tick within [start_ts, end_ts]
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    ticks = list(_iter_jsonl_ticks(str(p), "BTCUSDC", start_ts=500.0, end_ts=3000.0))
    assert ticks == [(1000.0, 100.0), (2000.0, 101.0)]


def test_incremental_trend_buffer_flat_recent_gradient_and_rebase():
    """IncrementalTrendBuffer matches instant_trend_from_prices on flat recent prices and after trim_front rebase."""
    from offline.research.monitortrades_backtest.replay_trend_source import _is_trend_up_tuple
    from pricewindow import IncrementalTrendBuffer, PriceWindow, instant_trend_from_prices

    buf = IncrementalTrendBuffer()
    prices = [111000.0 + i * 1.5 for i in range(600)] + [111702.29] * 5
    for px in prices:
        buf.append(px)

    res_inc = buf.eval_slice(0, len(prices) - 1, sample_rate_sec=15.0)
    res_ref = instant_trend_from_prices(prices, sample_rate_sec=15.0)
    assert res_inc[3] == 0.0
    assert res_ref[3] == 0.0
    assert _is_trend_up_tuple(res_inc[:4]) == _is_trend_up_tuple(res_ref)

    # Trigger rebase via trim_front (> 512 elements trimmed)
    buf.trim_front(520)
    trimmed = prices[520:]
    res_inc_after = buf.eval_slice(0, len(trimmed) - 1, sample_rate_sec=15.0)
    res_ref_after = instant_trend_from_prices(trimmed, sample_rate_sec=15.0)
    assert res_inc_after[3] == 0.0
    assert abs(res_inc_after[0] - res_ref_after[0]) < 1e-9
    assert _is_trend_up_tuple(res_inc_after[:4]) == _is_trend_up_tuple(res_ref_after)

    # Verify PriceWindow O(1) path matches reference
    pw = PriceWindow("TEST_INC", window_size=20, sample_rate_sec=15.0)
    for px in trimmed:
        pw.process_price(px)
    assert pw.get_short_trend()[3] == 0.0
    assert abs(pw.get_short_trend()[0] - instant_trend_from_prices(pw.prices, sample_rate_sec=15.0)[0]) < 1e-9


def test_smart_stop_loss_load_grace_state_sim_and_empty_file_caching(tmp_path, monkeypatch):
    """_load_grace_state skips disk I/O in simulation mode and caches empty {} state files by mtime."""
    import builtins
    import intelligence.external.guards.smart_stop_loss_guard as ssl_mod

    state_file = tmp_path / "smart_stop_loss_state.json"
    state_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ssl_mod, "_SMART_SL_STATE_FILE", str(state_file))
    monkeypatch.setattr(ssl_mod, "_GRACE_TRACKER", {})
    monkeypatch.setattr(ssl_mod, "_LOADED_STATE_FILE_KEY", None)

    open_calls = []
    real_open = builtins.open

    def counting_open(file, *args, **kwargs):
        if str(file) == str(state_file):
            open_calls.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    # Sim mode must never open the file
    ssl_mod._load_grace_state(is_sim=True)
    assert len(open_calls) == 0

    # Live mode opens once, then caches even when _GRACE_TRACKER remains {}
    ssl_mod._load_grace_state(is_sim=False)
    assert len(open_calls) == 1
    ssl_mod._load_grace_state(is_sim=False)
    ssl_mod._load_grace_state(is_sim=False)
    assert len(open_calls) == 1


def test_clear_position_stats_cache_respects_mt_persistence(tmp_path, monkeypatch):
    """clear_position_stats_cache does not delete _MT_STATE_FILE when persistence is disabled."""
    import monitortrades as mt

    state_file = tmp_path / "mt_state.json"
    state_file.write_text('{"BTCUSDC": {}}', encoding="utf-8")
    monkeypatch.setattr(mt, "_MT_STATE_FILE", str(state_file))
    monkeypatch.setattr(mt, "_MT_PERSISTENCE_ENABLED", False)

    mt.clear_position_stats_cache()
    assert state_file.exists()


def test_tradeall_backtest_window_constants_and_run(tmp_path, monkeypatch):
    """tradeall exports WINDOW_SECONDS_SMALL/BIG and offline.backtests.tradeall.run_backtest runs end-to-end."""
    import json
    import tradeall as ta
    from offline.backtests import tradeall as bt_ta

    assert hasattr(ta, "WINDOW_SECONDS_SMALL")
    assert hasattr(ta, "WINDOW_SECONDS_BIG")

    c24_file = tmp_path / "cache_24_price_ARBUSDC.json"
    items = [[1750000000000 + i * 15000, 0.20 + (i % 5) * 0.0005] for i in range(40)]
    c24_file.write_text(json.dumps({"items": {"ARBUSDC": items}}), encoding="utf-8")

    monkeypatch.setattr(bt_ta, "ROOT", str(tmp_path))
    bt_ta.run_backtest(
        symbol="ARBUSDC",
        start_ts=1750000000.0,
        end_ts=None,
        speed="fast",
        run_id="unit_test_run",
        source="cache24",
        cache24_file=str(c24_file),
        quiet=True,
    )
    pnl_path = tmp_path / "logger" / "backtest" / "unit_test_run" / "pnl.json"
    assert pnl_path.exists()


def test_backtest_suppresses_real_gemini_and_notifications(monkeypatch):
    """Backtest/replay modes strictly block real Gemini CLI/REST calls and external notifications."""
    import subprocess
    import urllib.request
    import monitortrades as mt
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard
    from notify_engine import alertnotifiers
    from strategies.position_exit_manager import ReentrySnapshot

    # 1. GeminiClient must not invoke subprocess.run or urlopen when BACKTEST_MODE/REPLAY_MODE/DISABLE_GEMINI_LLM is set
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Real CLI call attempted!"))
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Real REST call attempted!"))
    )
    client = GeminiClient(api_key="fake-key", cli_path=__file__)
    for env_key in ("BACKTEST_MODE", "REPLAY_MODE", "DISABLE_GEMINI_LLM"):
        monkeypatch.setenv(env_key, "1")
        assert client.query_text("test prompt") is None
        hs_dec = HighStakeGuard().check("BTCUSDC", "BUY", 50000.0, 1.0, notional_eur=50000.0)
        assert hs_dec.allowed is True
        assert hs_dec.reason == "backtest_mode_exempt"
        ps_dec = PostStopReentryGuard().check("BTCUSDC", 50000.0, 51000.0, 49000.0, 0.02)
        assert ps_dec.allowed is True
        assert ps_dec.reason == "backtest_mode_exempt"
        monkeypatch.delenv(env_key, raising=False)

    # 2. alertnotifiers.notify must be a no-op when BACKTEST_MODE or REPLAY_MODE is set
    dispatched = []
    monkeypatch.setenv("MPTRADE_ORCHESTRATED", "0")
    import notify_engine.server as srv_mod
    monkeypatch.setattr(
        srv_mod.NotificationServer, "dispatch_alerts", lambda self, alerts, **kw: dispatched.extend(alerts)
    )
    for env_key in ("BACKTEST_MODE", "REPLAY_MODE", "DISABLE_EXTERNAL_NOTIFICATIONS"):
        monkeypatch.setenv(env_key, "1")
        alertnotifiers.notify("Title", "Body", "backtest", "BTCUSDC")
        alertnotifiers.AlertNotifier.send_phone_webhook_batch([{"name": "test"}])
        alertnotifiers.AlertNotifier.send_email_batch([{"name": "test"}])
        assert dispatched == []
        monkeypatch.delenv(env_key, raising=False)

    # 3. monitortrades._apply_post_stop_ai_guard skips PostStopReentryGuard on simulated provider or disabled persistence
    snap = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price=100.0,
        last_sell_price=105.0,
        last_sell_time_s=1000.0,
        current_time_s=1500.0,
        last_exit_kind="STOP",
        post_sell_trough=98.0,
    )
    import intelligence.sentiment.guards.post_stop_reentry_guard as ps_mod
    monkeypatch.setattr(
        ps_mod,
        "get_post_stop_reentry_guard",
        lambda: (_ for _ in ()).throw(AssertionError("PostStopReentryGuard should be bypassed in replay!")),
    )

    class _ReplayInst:
        provider_name = "replay"

    allowed, bb, bq, q = mt._apply_post_stop_ai_guard(
        "BTCUSDC", 100.0, 105.0, 1000.0, snap, 100.0, None, 1.0, inst=_ReplayInst()
    )
    assert (allowed, bb, bq, q) == (True, 100.0, None, 1.0)


def test_kraken_continuous_grid_load_asset_prewarms_trend_series(tmp_path, monkeypatch):
    """kraken.replay exports prewarm_trend_series and kraken_continuous_grid.grid.load_asset succeeds over >WARM bars."""
    import offline.research.kraken_continuous_grid.grid as kgrid
    from kraken import replay as krp

    assert hasattr(krp, "prewarm_trend_series")
    csv_path = tmp_path / "TEST_240m.csv"
    lines = ["timestamp,open,high,low,close"]
    base_ts = 1_700_000_000
    for i in range(60):
        px = 100.0 + (i % 5) * 0.5
        lines.append(f"{base_ts + i * 14400},{px},{px + 1.0},{px - 1.0},{px}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(kgrid, "DATA", str(tmp_path))
    kgrid._LOAD_CACHE.clear()
    kgrid._ASSET_CACHE.clear()
    bars, ohlc, year_slices = kgrid.load_asset("TEST")
    assert len(bars) == 60
    assert len(ohlc) == 60
    assert isinstance(year_slices, list)


def test_tradeall_trigger_gate_common_isolation_and_price_window_kwarg(tmp_path, monkeypatch):
    """tradeall_trigger_gate.common isolates TrendState subclasses across variants and accepts price_window kwarg."""
    import tradeall as ta
    from offline.backtests import tradeall as bt_ta
    from offline.research.tradeall_trigger_gate import common as tg_common

    stats1 = {"starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0}
    stats2 = {"starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0}

    cls1 = tg_common.make_instrumented_trend_state(stats1)
    ta.TrendState = cls1
    try:
        cls2 = tg_common.make_instrumented_trend_state(stats2)
        assert issubclass(cls2, tg_common.ORIG_TREND_STATE)
        assert not issubclass(cls2, cls1)
        ts2 = cls2(3600, 160, 220)
        ts2.start_trend("UP")
        assert stats1["starts"] == 0
        assert stats2["starts"] == 1
    finally:
        ta.TrendState = tg_common.ORIG_TREND_STATE

    # Verify logic_variant accepts price_window keyword argument passed by ta.logic_small
    ts = tg_common.make_instrumented_trend_state(stats1)(3600, 160, 220)
    logic_fn = tg_common.make_tradeall_logic("VTEST", cooldown_mode="fire_once")
    logic_fn("SMALL", False, "ARBUSDC", 1, -1, ts, 1.0, price_window=None)
    assert stats1["starts"] == 1

    from offline.research.tradeall_trigger_gate import experiment_quality_signal as eqs
    eqs._trackers.clear()
    q_logic = eqs.make_quality_logic("VTEST_Q", 24, 1800)
    q_logic("BIG", False, "BTCUSDC", 1, -1, ts, 50000.0)
    assert "BTCUSDC" in eqs._trackers

    # Verify execute_tradeall_variant restores all patched tradeall attributes even after run
    c24_file = tmp_path / "cachedb" / "cache_365_price_ARBUSDC.jsonl"
    c24_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"s": "ARBUSDC", "i": [1750000000000 + i * 60000, 1.0 + i * 0.001]})
        for i in range(15)
    ]
    c24_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(tg_common, "ROOT", str(tmp_path))
    monkeypatch.setattr(bt_ta, "ROOT", str(tmp_path))

    pnl, _elapsed = tg_common.execute_tradeall_variant(
        "unit_tg_run",
        "ARBUSDC",
        1750000000.0,
        None,
        source="cache24",
        logic_fn=logic_fn,
        trend_state_cls=cls1,
        big_threshold=123.45,
        disable_small=True,
    )
    assert isinstance(pnl, dict)
    assert ta.TrendState is tg_common.ORIG_TREND_STATE
    assert ta.logic is tg_common.ORIG_LOGIC
    assert ta.PRICE_CHANGE_THRESHOLD_BIG_EUR == tg_common.ORIG_BIG_THRESHOLD


def test_research_scripts_do_not_monkeypatch_at_import_time():
    """Importing research grid/experiment scripts must not leave Strategy.step or dca_price_hit mutated."""
    import strategies.spot_engine as spot_engine
    import strategies.spot_rules as spot_rules
    import offline.research.kraken_continuous_grid.grid as kgrid
    import offline.research.tradeall_trigger_gate.experiment_rsi_dca_filter as rsi_dca

    assert spot_engine.Strategy.step is kgrid._orig_step
    assert spot_engine.Strategy.step is rsi_dca._orig_step
    assert spot_rules.dca_price_hit is rsi_dca._orig_hit


def test_intelligence_and_chop_backtest_fee_accounting_reconciliation(tmp_path):
    """Per-trade PnL in intelligence_backtest and chop_weight_backtest includes buy fees and reconciles with net_profit_usd."""
    import math
    from offline.research.chop_weight_backtest import run_single_simulation
    from offline.research.intelligence_backtest import run_intelligence_backtest

    # Create synthetic oscillating price series that triggers trades
    sampled = []
    t0 = 1_750_000_000.0
    for i in range(160):
        wave = math.sin(i / 6.0) * 8.0
        drift = i * 0.15
        sampled.append((t0 + i * 900.0, 100.0 + drift + wave))

    stat = run_single_simulation(
        symbol="TAOUSDC",
        sampled=sampled,
        btc_sampled=sampled,
        policy="CHOP_0.10",
        step_seconds=900,
        initial_balance=10_000.0,
        fee_pct=0.005,  # 0.5% fee to make any missing buy-fee discrepancy obvious
    )
    assert stat.total_trades >= 1
    assert math.isclose(
        stat.chop_pnl_usd + stat.trend_pnl_usd,
        stat.net_profit_usd,
        abs_tol=0.02,
    )

    cache_file = tmp_path / "cache_sparse_price_TESTUSDC.jsonl"
    lines = [
        json.dumps({"s": "TESTUSDC", "i": [int(t * 1000), p]})
        for t, p in sampled
    ]
    cache_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ib_res = run_intelligence_backtest(
        symbol="TESTUSDC",
        cache_path=str(cache_file),
        step_seconds=900,
        initial_balance=10_000.0,
        fee_pct=0.005,
    )
    for mode_stat in ib_res.values():
        assert math.isclose(
            mode_stat.final_balance - mode_stat.initial_balance,
            mode_stat.net_profit_usd,
            abs_tol=0.02,
        )


def test_simulations_price_window_get_max_and_normalize_indices():
    """offline.simulations.sim.PriceWindow.get_max returns true sliding-window maximum and sim2 reuses it."""
    from offline.simulations import sim, sim2

    assert sim2.PriceWindow is sim.PriceWindow
    pw = sim.PriceWindow(window_size=4, max_index=5)
    for px in (10.0, 50.0, 30.0, 20.0):
        pw.process_price(px)
    assert pw.get_min()[0] == 10.0
    assert pw.get_max()[0] == 50.0

    # Advance past 50.0 so the window holds (30.0, 20.0, 25.0, 15.0) and triggers normalize_indices
    pw.process_price(25.0)
    pw.process_price(15.0)
    assert pw.get_min()[0] == 15.0
    assert pw.get_max()[0] == 30.0

    # Non-finite, bool, and None values are ignored without poisoning min/max deques
    pw.process_price(float("nan"))
    pw.process_price(float("inf"))
    pw.process_price(True)
    pw.process_price(None)
    assert pw.get_min()[0] == 15.0
    assert pw.get_max()[0] == 30.0


def test_experiment_trend_gate_and_kalman_lag_sweep_restore_globals(tmp_path, monkeypatch):
    """experiment_trend_gate and run_kalman_lag_sweep restore tradeall/shadow_signals globals after execution."""
    import shadow_signals
    import tradeall as ta
    from offline.backtests import tradeall as bt_ta
    from offline.research.tradeall_kalman_lag import run_kalman_lag_sweep as kal_sweep
    from offline.research.tradeall_trigger_gate import common as tg_common
    from offline.research.tradeall_trigger_gate import experiment_trend_gate as etg

    cls = etg.make_instrumented(
        "UNIT_ETG",
        confirm_threshold_override=2,
        expiration_trend_time_override=9999,
    )
    stats = etg.STATS["UNIT_ETG"]
    ts = cls(3600, 160, 220)
    assert ts.expiration_trend_time == 9999
    ts.start_trend("UP")
    for _ in range(5):
        ts.confirm_trend()
    assert stats["confirms"] == 5
    assert stats["max_confirm_count"] == 6

    c24_file = tmp_path / "cachedb" / "cache_365_price_ARBUSDC.jsonl"
    hist_file = tmp_path / "cachedb" / "cache_sparse_price_ARBUSDC.jsonl"
    c24_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"s": "ARBUSDC", "i": [1750000000000 + i * 60000, 1.0 + i * 0.001]})
        for i in range(10)
    ]
    payload = "\n".join(rows) + "\n"
    c24_file.write_text(payload, encoding="utf-8")
    hist_file.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(tg_common, "ROOT", str(tmp_path))
    monkeypatch.setattr(bt_ta, "ROOT", str(tmp_path))
    monkeypatch.setattr(kal_sweep, "ROOT", str(tmp_path))

    orig_old = ta.TREND_TO_BE_OLD_SECONDS
    tg_common.execute_tradeall_variant(
        "unit_old_override",
        "ARBUSDC",
        1750000000.0,
        None,
        source="cache24",
        trend_state_cls=cls,
        trend_to_be_old_override=1800,
    )
    assert ta.TREND_TO_BE_OLD_SECONDS == orig_old

    orig_kalman_sec = shadow_signals.KALMAN_SAMPLE_SEC
    res = kal_sweep.run_variant("ARBUSDC", 1750000000.0, None, 120.0, "unit_kal_run")
    assert shadow_signals.KALMAN_SAMPLE_SEC == orig_kalman_sec
    assert res["sample_sec"] == 120.0


def test_align_spot_interval_params_and_walk_forward_compare_multi_interval():
    """align_spot_interval_params aligns regime/volatility intervals so multi-interval walk-forward evaluation succeeds."""
    from offline.runners.walk_forward_compare import _segment_result
    from strategies.spot_engine import StratParams
    from strategies.spot_replay import align_spot_interval_params

    base = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.5,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        check_minutes=2.0,
        takeprofit_pct=3.0,
        max_budget=500.0,
        max_dca_buys=3,
        enable_takeprofit=True,
        order_ttl_min=10.0,
        stop_loss_pct=15.0,
        trend_overlay=True,
        reentry_hybrid_enabled=True,
        tp_trail_adaptive=True,
        dca_vol_scale_k=1.0,
        trend_interval=240,
        tp_trail_vol_interval=240,
        dca_vol_interval=240,
    )
    aligned_60 = align_spot_interval_params(base, 60)
    assert aligned_60.trend_interval == 60
    assert aligned_60.tp_trail_vol_interval == 60
    assert aligned_60.dca_vol_interval == 60
    aligned_1440 = align_spot_interval_params(base, 1440)
    assert aligned_1440.trend_interval == 1440
    assert aligned_1440.regime_min_samples <= 7
    assert align_spot_interval_params(base, 240) is base

    records = [
        {"timestamp": 1_700_000_000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(10)
    ]
    seg_60 = _segment_result(records, base, fee_pct=0.26, interval=60)
    seg_1440 = _segment_result(records, base, fee_pct=0.26, interval=1440)
    assert seg_60["metrics"]["account_currency"] == "USD"
    assert seg_1440["metrics"]["account_currency"] == "USD"


def test_hybrid_reentry_equities_base_config_and_backtest_invariants(tmp_path):
    """Verify run_equities BASE_CONFIG, BacktestBroker unbacked SELL guard, InstrumentedTrendState super call, and scheduled_pilot diag cache isolation."""
    from offline.backtests.tradeall import BacktestBroker
    from offline.research.hybrid_reentry import run_equities
    from offline.research.monitortrades_backtest import run_replay_backtest as rb
    from offline.research.monitortrades_backtest import scheduled_pilot as sp
    from offline.research.tradeall_trigger_gate import common as tg_common

    # 1. run_equities.BASE_CONFIG must satisfy 212trading StratParams.from_env
    strat_mod, _rp = run_equities._import_t212()
    cfg = dict(run_equities.BASE_CONFIG, YAHOO_SYMBOL="NVDA")
    params = strat_mod.StratParams.from_env(cfg)
    assert params.yahoo_sym == "NVDA"
    assert params.reentry_pullback_adaptive is False

    # 2. BacktestBroker.place_order_smart must reject unbacked SELL when flat even if trend_gate wants to wait
    class _AlwaysWaitGate:
        def should_wait(self, symbol, side, timeout_sec):
            return True

    broker = BacktestBroker(str(tmp_path), clock=lambda: 1000.0, trend_gate=_AlwaysWaitGate())
    assert broker.place_order_smart("SELL", "BTCUSDC", 50000.0) is None
    assert broker.n_waited == 0
    assert broker.pending == []

    # 3. InstrumentedTrendState.mark_confirmed must increment parent TrendState confirmation counter
    stats = {"starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0}
    ts_cls = tg_common.make_instrumented_trend_state(stats)
    ts_inst = ts_cls(3600, 160, 220)
    ts_inst.start_trend("UP")
    ts_inst.mark_confirmed("UP")
    assert ts_inst.already_confirmed("UP") is True
    assert ts_inst._confirmed_count_up == 1

    # 4. scheduled_pilot._RUN_ONE_DIAG_CACHE isolates diagnostics per (symbol, params, series)
    series = [(1_700_000_000.0 + i * 60.0, 100.0 + i * 0.1) for i in range(105)]
    p_a = {"mt.hardtp": "3.0"}
    p_b = {"mt.hardtp": "20.0"}
    ck_a = sp._run_one_cache_key("BTCUSDC", p_a, series)
    ck_b = sp._run_one_cache_key("BTCUSDC", p_b, series)
    tk = rb._series_cache_key("BTCUSDC", series, rb.DEFAULT_WINDOW_SECONDS)
    sp._RUN_ONE_CACHE[ck_a] = {"net": 15.0, "buy_hold": 5.0}
    sp._RUN_ONE_DIAG_CACHE[ck_a] = {"hard_tp_touched": True, "min_hard_tp_gain_ratio": 0.035}
    sp._RUN_ONE_CACHE[ck_b] = {"net": 10.0, "buy_hold": 5.0}
    sp._RUN_ONE_DIAG_CACHE[ck_b] = {"hard_tp_touched": False, "min_hard_tp_gain_ratio": 0.0}
    # Simulate worker B finishing last and overwriting params-agnostic rb._LAST_RUN_DIAGNOSTICS[tk]
    rb._LAST_RUN_DIAGNOSTICS[tk] = dict(sp._RUN_ONE_DIAG_CACHE[ck_b])
    rb._LAST_HARD_TP_TOUCHED[tk] = False
    # Calling _run_one for p_a must restore p_a's own diagnostics
    res_a = sp._run_one("BTCUSDC", "BTC", p_a, series)
    assert res_a == {"net": 15.0, "buy_hold": 5.0}
    assert rb._LAST_HARD_TP_TOUCHED[tk] is True
    assert rb._LAST_RUN_DIAGNOSTICS[tk]["min_hard_tp_gain_ratio"] == 0.035

    # 5. migrate_price_cache outputs cache_sparse_price_<SYM>.jsonl
    from offline.legacy_tools import migrate_price_cache as mpc
    legacy_json = tmp_path / "cache_price_TESTUSDC.json"
    legacy_json.write_text(json.dumps({"items": {"TESTUSDC": [[1700000000000, 123.45]]}}), encoding="utf-8")
    added = mpc.migrate_file(str(legacy_json), dry_run=False)
    assert added == 1
    assert (tmp_path / "cache_sparse_price_TESTUSDC.jsonl").exists()
    assert (tmp_path / "cache_price_TESTUSDC.json.bak").exists()

    # 6. binance_walk_forward_baseline.main runs end-to-end with validate_dataset
    from unittest.mock import patch
    from offline.runners import binance_walk_forward_baseline as bwf
    fake_records = [
        {"timestamp": 1700000000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(30)
    ]
    with patch.object(bwf, "fetch_ohlc_candles", return_value=fake_records):
        rc = bwf.main(["--intervals", "60", "--symbol", "BTCUSDC", "--output-dir", str(tmp_path / "bwf")])
        assert rc == 0
        assert (tmp_path / "bwf" / "BTCUSDC_60m_baseline.json").exists()


def test_offline_research_ml_and_pilot_edge_cases(tmp_path, monkeypatch):
    """Verify edge-case guards in scheduled_pilot, slice_dense_archive, forecast, priceprediction, vol_chronos, and hybrid_reentry.harness."""
    import numpy as np
    from offline.research import slice_dense_archive as sda
    from offline.research.hybrid_reentry import harness as hh
    from offline.research.ml_forecast import forecast as ml_fc
    from offline.research.ml_forecast import priceprediction as ml_pp
    from offline.research.ml_forecast import vol_chronos as ml_vc
    from offline.research.monitortrades_backtest import scheduled_pilot as sp

    # 1. scheduled_pilot._last_change_for ignores non-dict JSON lines
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        '123\n["list"]\n{"full_key": "BINANCE_BTC.mt.gain", "action": "applied", "ts": "2026-10-26T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sp, "AUDIT_LOG", str(audit_path))
    last = sp._last_change_for("BINANCE_BTC.mt.gain")
    assert isinstance(last, dict)
    assert last["ts"] == "2026-10-26T00:00:00+00:00"

    # 2. slice_dense_archive.ts_of handles non-dict, empty list "i", bool, and NaN
    assert sda.ts_of(None) == 0.0
    assert sda.ts_of([1, 2]) == 0.0
    assert sda.ts_of({"i": []}) == 0.0
    assert sda.ts_of({"i": [True]}) == 0.0
    assert sda.ts_of({"i": [float("nan")]}) == 0.0
    assert sda.ts_of({"i": [], "t": 1700000000000}) == 1700000000000.0

    # 3. ml_forecast.forecast.walk_forward clamps refit_every<=0 without infinite loop
    X = np.random.RandomState(0).randn(20, 11)
    y_dir = np.array([i % 2 for i in range(20)])
    y_mag = np.linspace(-0.02, 0.02, 20)
    rep = ml_fc.walk_forward(X, y_dir, y_mag, train_frac=0.7, refit_every=0)
    assert rep["n_test"] == 6
    assert np.isfinite(rep["boost"])

    # 4. ml_forecast.priceprediction.PricePrediction rejects invalid prices and clamps window_size
    pp = ml_pp.PricePrediction(window_size=0, epochs=0)
    assert pp.window_size == 2
    assert pp.epochs == 1
    pp.process_price(float("nan"))
    pp.process_price(float("inf"))
    pp.process_price(-10.0)
    pp.process_price(True)  # type: ignore[arg-type]
    assert pp.prices == []
    pp.process_prices([100.0, float("nan"), 101.0, 0.0, 102.0, 103.0])
    assert pp.prices == [100.0, 101.0, 102.0, 103.0]
    assert pp.trained is True

    # 5. ml_forecast.vol_chronos.walk_forward_vol guards stride<=0 and empty window range
    rv = np.ones(30, dtype=float)
    vc_rep = ml_vc.walk_forward_vol(rv, horizon=24, warmup=20, stride=0)
    assert vc_rep["n_test"] == 0
    assert vc_rep["improvement_pct"] is None

    # 6. hybrid_reentry.harness.load_ohlc_csv preserves headerless first row and filters NaN
    csv_no_hdr = tmp_path / "no_hdr.csv"
    csv_no_hdr.write_text(
        "1700000000,100.0,105.0,99.0,102.0\n1700003600,nan,106.0,98.0,101.0\n1700007200,102.0,107.0,100.0,104.0\n",
        encoding="utf-8",
    )
    rows = hh.load_ohlc_csv(str(csv_no_hdr))
    assert len(rows) == 2
    assert rows[0] == (100.0, 105.0, 99.0, 102.0)

    row_fmt = hh.format_ablation_result(
        "BTC", "Test 0", "Baseline",
        {"total": None, "cycles": None, "wins": None, "fills": None},
        capital=1000.0,
    )
    assert row_fmt["net_profit_usd"] == 0.0
    assert row_fmt["cycles"] == 0
    assert row_fmt["fills"] == 0


def test_ui_mirror_hot_reload_and_research_sweep_guards(tmp_path, monkeypatch):
    """Verify ui_mirror._safe_step_count fallback and comprehensive_strategy_sweep / kraken_adaptive_thresholds guards."""
    from orchestratorOS.agy import ui_mirror
    from offline.research import comprehensive_strategy_sweep as css
    from offline.research.kraken_adaptive_thresholds import verify_adaptive_dca, verify_adaptive_reentry

    # 1. ui_mirror._fallback_safe_step_count handles bool, NaN, negative, and valid floats
    assert ui_mirror._fallback_safe_step_count(None) == 0
    assert ui_mirror._fallback_safe_step_count(True) == 0
    assert ui_mirror._fallback_safe_step_count(float("nan")) == 0
    assert ui_mirror._fallback_safe_step_count("-4") == 0
    assert ui_mirror._fallback_safe_step_count("7.0") == 7

    # 2. comprehensive_strategy_sweep._summarize_replay coalesces None values
    summary = css._summarize_replay(
        {"net": None, "total": None, "max_drawdown_pct": None, "cycles": None, "wins": None, "fills": None},
        budget=1000.0,
    )
    assert summary["net_pct"] == 0.0
    assert summary["total_pct"] == 0.0
    assert summary["cycles"] == 0
    assert summary["wins"] == 0
    assert summary["fills"] == 0

    # 3. verify_adaptive_dca and verify_adaptive_reentry handle empty candles cleanly
    monkeypatch.setattr(verify_adaptive_dca, "fetch_candles", lambda pair, interval: [])
    monkeypatch.setattr(verify_adaptive_reentry, "fetch_candles", lambda pair, interval: [])
    assert verify_adaptive_dca.run() == ({}, {})
    assert verify_adaptive_reentry.run() == ({}, {})

    # 4. kraken_continuous_grid report/report2/report3 and t212 report handle malformed JSONL and empty errors
    from offline.research.kraken_continuous_grid import report as kr_rep1, report2 as kr_rep2, report3 as kr_rep3
    (tmp_path / "results.jsonl").write_text(
        '123\nnot_json\n{"asset": "HYPE", "cfg": "live", "error": ""}\n'
        '{"asset": "HYPE", "cfg": "live", "full": {"ret": 5.0, "dd": 2.0, "expo": 50.0, "bh": 3.0}, "w90": null, "w30": [[1.0, 0.5, null]]}\n',
        encoding="utf-8",
    )
    (tmp_path / "results2.jsonl").write_text(
        'null\n{"kind": "error", "args": ["views", "HYPE"], "error": null}\n',
        encoding="utf-8",
    )
    (tmp_path / "results3.jsonl").write_text(
        '"str"\n{"asset": "HYPE", "cfg": "live", "error": "   "}\n',
        encoding="utf-8",
    )
    assert any("unknown" in line for line in kr_rep1.generate_report(str(tmp_path)))
    assert any("unknown" in line for line in kr_rep2.generate_report(str(tmp_path)))
    assert any("unknown" in line for line in kr_rep3.generate_report(str(tmp_path)))
    t212_out = t212_report.generate_report(str(tmp_path))
    assert "unknown" in t212_out

    # 5. tradeall_trigger_gate LongTrendTracker, RsiGate, MeanReversionTracker, and STATS reset per run
    from offline.research.tradeall_trigger_gate import (
        common as tg_common,
        experiment_cooldown_only_and_tighten as exp7,
        experiment_quality_signal as exp5,
        experiment_quality_signal_v2 as exp6,
        experiment_rsi_bollinger as exp8,
        experiment_rsi_dca_filter as exp9,
    )
    lt_stats = {}
    lt = tg_common.LongTrendTracker(lt_stats, window_hours=0, reeval_sec=0, min_points=1)
    assert lt.window_sec in (60.0, 3600.0)
    assert lt.reeval_sec in (1.0, 60.0)
    assert lt.min_points in (2, 30)
    assert lt.update(float("nan"), 100.0) == 0
    assert lt.update(1000.0, float("nan")) == 0
    assert lt.update(1000.0, -5.0) == 0
    assert len(lt.buf) == 0

    _, st7_a = exp7.make_instrumented("tag7")
    st7_a["starts"] = 5
    _, st7_b = exp7.make_instrumented("tag7")
    assert st7_b["starts"] == 0

    exp5.make_quality_logic("tag5", 24, 1800)
    exp5.STATS["tag5"]["sign_changes"] = 4
    exp5.make_quality_logic("tag5", 24, 1800)
    assert exp5.STATS["tag5"]["sign_changes"] == 0

    st6_a = exp6._make_stats("tag6")
    st6_a["sign_changes"] = 3
    st6_b = exp6._make_stats("tag6")
    assert st6_b["sign_changes"] == 0

    exp8._make_stats("tag8")
    mrt = exp8.MeanReversionTracker("tag8", "BTCUSDC", mode="both", rsi_period=0, bb_period=0)
    assert mrt.rsi_period == 1
    assert mrt.bb_period == 1
    assert mrt.update_sign(float("nan")) == 0

    rg = exp9.RsiGate(0, 30.0)
    assert rg.period == 1
    rg.update(float("nan"))
    assert rg.prev is None

    # 6. clean_order_cache, migrate_price_cache, and verify_account_facade edge cases
    from offline.legacy_tools import clean_order_cache as coc, migrate_price_cache as mpc
    from verify_tools import verify_account_facade as vaf

    assert coc.is_garbage(None) is True
    assert coc.is_garbage({"status": "FILLED", "price": float("nan")}) is True
    assert coc.is_garbage({"status": "FILLED", "price": float("inf")}) is True
    assert coc.is_garbage({"status": "FILLED", "price": True}) is True
    assert coc.is_garbage({"status": "FILLED", "price": 100.0}) is False

    assert mpc._entry_ts([None, 100.0]) == 0
    assert mpc._entry_ts([True, 100.0]) == 0
    assert mpc._entry_ts([float("nan"), 100.0]) == 0
    assert mpc._entry_ts(["1700000000000", 100.0]) == 1700000000000

    assert vaf.order_key(None) == (0.0, 0.0, 0.0, "")
    orders_mixed = [
        {"timestamp": None, "price": "100.0", "qty": "1.0", "side": "buy"},
        {"timestamp": 1700000000, "price": 99.0, "qty": 2.0, "side": "BUY"},
    ]
    sorted_orders = sorted(orders_mixed, key=vaf.order_key)
    assert len(sorted_orders) == 2
    avg_p, net_q = vaf.position_from(orders_mixed, [{"qty": "0.5", "price": 105.0}])
    assert net_q == 2.5
    assert avg_p > 0

    # 7. replay_trend_source, run_deep_surge_sweep, and compare_mhdsl_reentry_replay edge cases
    from offline.research import compare_mhdsl_reentry_replay as cmr
    from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as rdss
    from offline.research.monitortrades_backtest import replay_trend_source as rts

    rts_inst = rts.ReplayTrendSource(["BTCUSDC"], window_seconds=-10)
    assert rts_inst.window_seconds == rts.DEFAULT_WINDOW_SECONDS
    rts_inst.advance("BTCUSDC", float("nan"), 100.0)
    rts_inst.advance("BTCUSDC", 1000.0, -5.0)
    assert len(rts_inst._windows["BTCUSDC"].prices) == 0

    drts = rts.DynamicReplayTrendSource(["BTCUSDC"], max_timeout_sec=0)
    assert drts.max_timeout_sec == rts.SMALL_TIMEOUT_RANGE_SEC[1]
    drts.advance("BTCUSDC", float("nan"), 100.0)
    assert len(drts._buf["BTCUSDC"]) == 0
    assert drts.is_trend_up_confirmed("BTCUSDC", []) is False

    assert rdss._win_move_at_ts([100.0, 110.0], 0, float("nan"), 4.0) is None
    assert rdss._win_move_at_ts([100.0, 110.0], 0, 240.0 * 60.0, 0.0) is None
    assert rdss._win_move_at_ts([0.0, 110.0], 0, 240.0 * 60.0, 4.0) is None
    assert rdss._win_move_at_ts([100.0, 110.0], 0, 240.0 * 60.0, 4.0) == 10.0

    empty_csv = tmp_path / "empty_hype.csv"
    empty_csv.write_text("timestamp,open,high,low,close\n1700000000,nan,10,9,10\n", encoding="utf-8")
    assert cmr.run_comparison(str(empty_csv)) == []

    # 8. backtest_ranges and legacy move.backup_and_replace edge cases
    from offline.legacy_tools import move as legacy_move
    from offline.research import backtest_ranges as br

    assert br.scan_backtest_ranges(str(tmp_path)) == {}
    cfg_file = tmp_path / "ranges.conf"
    cfg_file.write_text(
        "[BINANCE_BTC]\n# BACKTEST: , , \nmt.gain = 7.0\n# BACKTEST: 5.0, 6.0\nmt.lost = 3.0\n",
        encoding="utf-8",
    )
    assert br.scan_backtest_ranges(str(cfg_file)) == {"BINANCE_BTC.mt.lost": ["5.0", "6.0"]}

    cache_orig = tmp_path / "cache_price_TEST.json"
    cache_orig.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        legacy_move.backup_and_replace(str(tmp_path / "missing_converted.json"), str(cache_orig))
    assert cache_orig.exists()

    # 9. spot_profile_review.revision_params filters None values from dotenv_values
    from types import SimpleNamespace
    from unittest.mock import patch
    from offline.runners import spot_profile_review as spr
    cfg_text = (spr.ROOT / "hyperliquid/config.env").read_text(encoding="utf-8") + "\nUNVALUED_KEY\n"
    with patch.object(spr.subprocess, "run", return_value=SimpleNamespace(stdout=cfg_text)):
        rp = spr.revision_params("HEAD:hyperliquid/config.env")
        assert rp.entry_amount > 0

    # 10. assess_decision_evidence, calibrate_execution_events, walk_forward.run_compare, tradeall_observe, drawprices
    from offline.backtests.evaluation import assess_decision_evidence
    from offline.backtests.execution_calibration import calibrate_execution_events
    from offline.runners import walk_forward as wf_disp
    from visuals import drawprices as dp, tradeall_observe as tob

    ev = assess_decision_evidence(
        [{"timestamp": 1700000000}, {"timestamp": 1700086400}],
        {"fold_count": None, "total_cycles": None, "worst_return_pct": None},
    )
    assert ev["decision_grade_data"] is False

    cal = calibrate_execution_events([
        {"intent_id": "i1", "event": "submit_requested", "venue": "kraken", "symbol": "HYPEUSD", "qty": 1.0},
        {"intent_id": "i1", "event": "submit_accepted", "venue": "kraken", "symbol": "HYPEUSD", "accepted_ts": "bad_ts", "ts": "bad_ts", "client_order_id": "a" * 32, "order_id": "o1"},
    ])
    assert cal["client_order_id_validation"]["valid_client_order_ids"] == 1

    with patch("offline.runners.walk_forward_compare.main", return_value=0) as mock_wfc:
        assert wf_disp.run_compare(["--venue=kraken", "rep.json"], venue="kraken") == 0
        mock_wfc.assert_called_once_with(["--venue=kraken", "rep.json"])

    X_empty, y_empty = dp.make_sequences(dp.np.array([1.0]), window=0)
    assert X_empty.shape == (0, 1, 1)
    assert y_empty.shape == (0,)

    c_dir = tmp_path / "cachedb"
    c_dir.mkdir(exist_ok=True)
    (c_dir / "cache_365_price_TEST.jsonl").write_text('123\n[1,2]\n{"s": "TEST", "i": [1000, 50.0]}\n', encoding="utf-8")
    (c_dir / "cache_24_price_TEST.json").write_text('{"items": []}', encoding="utf-8")
    monkeypatch.setattr(tob, "ROOT", str(tmp_path))
    assert tob._load_cachedb_price_entries("cache_365_price_TEST.jsonl", "TEST") == [[1000, 50.0]]
    assert tob._load_cachedb_price_entries("cache_24_price_TEST.json", "TEST") == []

    # 11. analyze_volatility.pullbacks empty input, 212trading/backtest.simulate None/shorthand keys, hyperliquid/backtest.precompute_signal
    import importlib.util
    import sys
    from pathlib import Path
    root_dir = Path(tob.__file__).resolve().parent.parent

    with monkeypatch.context() as m:
        m.setattr(sys, "path", list(sys.path))
        spec_av = importlib.util.spec_from_file_location("t212_analyze_vol_test", root_dir / "212trading/analyze_volatility.py")
        mod_av = importlib.util.module_from_spec(spec_av)
        spec_av.loader.exec_module(mod_av)
        assert mod_av.pullbacks([]) == []
        assert mod_av.pullbacks([100.0, 80.0, 102.0]) == [25.0]

        spec_t212_bt = importlib.util.spec_from_file_location("t212_backtest_test", root_dir / "212trading/backtest.py")
        mod_t212_bt = importlib.util.module_from_spec(spec_t212_bt)
        spec_t212_bt.loader.exec_module(mod_t212_bt)
        sim_res = mod_t212_bt.simulate(
            [(100.0, 102.0, 99.0, 101.0), (101.0, 103.0, 100.0, 102.0)],
            {
                "budget": 1000.0,
                "entry": 100.0,
                "dca": 100.0,
                "disc": 0.2,
                "drop": 2.0,
                "tp": 1.5,
                "maxdca": 5,
                "fee": 0.15,
                "sl": 10.0,
                "tp_ladder": None,
                "reentry_drop": 1.0,
                "reentry_tol": 0.05,
                "sl_rebuy_bounce": 1.2,
                "sl_mhdsl_enabled": True,
            },
        )
        assert isinstance(sim_res, dict) and "total" in sim_res

        spec_hl_bt = importlib.util.spec_from_file_location("hl_backtest_test", root_dir / "hyperliquid/backtest.py")
        mod_hl_bt = importlib.util.module_from_spec(spec_hl_bt)
        spec_hl_bt.loader.exec_module(mod_hl_bt)
        sig_out = mod_hl_bt.precompute_signal([100.0, 101.0, 102.0], [1000.0, 4600.0, 8200.0], "analysis", 2, 3, 0.01, 24, 8)
        assert sig_out == ["neutral", "neutral", "neutral"]


def test_scheduled_pilot_and_cooldown_config_parsing_and_hl_signals_fallback(tmp_path, monkeypatch):
    """Verify inline comment/interpolation handling in scheduled_pilot, run_replay_backtest, trade_cooldown, and hl signals fallback."""
    import importlib.util
    import sys
    from pathlib import Path

    from offline.research.monitortrades_backtest import run_replay_backtest as rrb
    from offline.research.monitortrades_backtest import scheduled_pilot as sp
    import lock.trade_cooldown as tc

    conf_file = tmp_path / "instruments.conf"
    conf_file.write_text(
        "[BTCEUR] ; primary section comment with 50% note\n"
        "mt.gain = 7.5 ; semicolon comment with 10% note\n"
        "mt.lost = 3.2 # hash comment with 20% note\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sp, "INSTRUMENTS_CONF", str(conf_file))
    assert sp._current_value("BTCEUR", "mt.gain") == 7.5
    assert sp._current_value("BTCEUR", "mt.lost") == 3.2

    sp._apply_config_change("BTCEUR", "mt.gain", 7.5, 8.25)
    updated_text = conf_file.read_text(encoding="utf-8")
    assert "mt.gain = 8.25 ; semicolon comment with 10% note" in updated_text
    assert sp._current_value("BTCEUR", "mt.gain") == 8.25

    monkeypatch.setattr(rrb, "ROOT", str(tmp_path))
    live_params = rrb._live_mt_params("BTCEUR")
    assert live_params == {"mt.gain": "8.25", "mt.lost": "3.2"}

    cd_conf = tmp_path / "trade_cooldown.conf"
    cd_conf.write_text("[cooldown]\ndefault_sec = 7 ; inline comment with 100% safety\n", encoding="utf-8")
    monkeypatch.setattr(tc, "CONF_FILE", str(cd_conf))
    assert tc._load_cooldown_sec() == 7

    root_dir = Path(sp.__file__).resolve().parents[3]
    with monkeypatch.context() as m:
        m.setattr(sys, "path", [p for p in sys.path if "hyperliquid" not in p])
        spec_sig = importlib.util.spec_from_file_location("hl_signals_test", root_dir / "hyperliquid/signals.py")
        mod_sig = importlib.util.module_from_spec(spec_sig)
        spec_sig.loader.exec_module(mod_sig)
        # Remove hyperliquid dir from sys.path after import so `from price_analysis import signal` fails
        m.setattr(sys, "path", [p for p in sys.path if "hyperliquid" not in p])
        m.syspath_prepend(str(tmp_path))
        m.setenv("SIGNAL_SOURCE", "analysis")
        m.setenv("TREND_WINDOW_HOURS", "24")
        m.setenv("TREND_RESAMPLE_HOURS", "8")

        class _DummyClient:
            def candles(self, coin, interval, lookback_hours=30):
                return []

        res = mod_sig.get_signal(_DummyClient(), "BTC")
        assert res["trend"] == "neutral"


def test_hl_price_analysis_mid_and_strategy_run_and_backtest_ranges_inline_comment(tmp_path, monkeypatch):
    """Verify HL price_analysis non-numeric mid safety, HL Strategy.run exception resilience, backtest_ranges section comments, and chop_benchmark step validation."""
    import importlib.util
    import sys
    from pathlib import Path
    from unittest.mock import patch

    from offline.research import backtest_ranges as br
    from offline.research import chop_weight_backtest as cwb
    import market_monitor.pricefetcher as pf

    # 1. backtest_ranges section header with inline comment
    conf_p = tmp_path / "ranges.conf"
    conf_p.write_text(
        "[BINANCE_BTC] ; inline comment\n"
        "# BACKTEST: 5.0, 6.0, 7.0\n"
        "mt.gain = 6.0\n",
        encoding="utf-8",
    )
    assert br.scan_backtest_ranges(str(conf_p)) == {"BINANCE_BTC.mt.gain": ["5.0", "6.0", "7.0"]}

    # 2. chop_weight_backtest.run_chop_benchmark rejects invalid step_seconds immediately
    with pytest.raises(ValueError, match="step_seconds"):
        cwb.run_chop_benchmark(symbols=[], step_seconds=0)
    with pytest.raises(ValueError, match="step_seconds"):
        cwb.run_chop_benchmark(symbols=[], step_seconds=True)

    # 3. pricefetcher.get_price_history formats both seconds and ms timestamps accurately
    with patch.object(pf, "PricePlatformFactory"):
        pf_mgr = pf.CacheAllPriceFetcherManager(60, ["BTC"], str(tmp_path / "pf.json"), read_only=True)
    pf_mgr.cache["BTC"] = [[1700000000, 50000.0], [1700000000000, 50100.0]]
    hist = pf_mgr.get_price_history("BTC", limit=5)
    assert len(hist) == 2
    assert hist[0]["timestamp_readable"].startswith("2023-11-")
    assert hist[1]["timestamp_readable"].startswith("2023-11-")

    # 4. hyperliquid/price_analysis.signal handles non-numeric client.mid() without ValueError
    root_dir = Path(br.__file__).resolve().parents[2]
    with monkeypatch.context() as m:
        m.setattr(sys, "path", list(sys.path))
        spec_pa = importlib.util.spec_from_file_location("hl_pa_mid_test", root_dir / "hyperliquid/price_analysis.py")
        mod_pa = importlib.util.module_from_spec(spec_pa)
        spec_pa.loader.exec_module(mod_pa)
        m.setattr(
            mod_pa,
            "analyze",
            lambda _c, _coin: {
                "direction": "up",
                "start_timestamp": 1000.0,
                "duration_seconds": 36000.0,
                "estimated_future_hours": 5.0,
                "current_slope_h": 0.5,
            },
        )

        class _BadMidClient:
            def mid(self, _coin):
                return "N/A"

        sig = mod_pa.signal(_BadMidClient(), "HYPE")
        assert sig["trend"] == "up"
        assert sig["source"] == "analysis"

        # 5. hyperliquid/strategy.py _check_stop_loss keeps uncancelled orders tracked & run() survives step error
        spec_st = importlib.util.spec_from_file_location("hl_strat_run_test", root_dir / "hyperliquid/strategy.py")
        mod_st = importlib.util.module_from_spec(spec_st)
        m.setitem(sys.modules, spec_st.name, mod_st)
        spec_st.loader.exec_module(mod_st)

        class _FailCancelClient:
            exchange = None

            def sz_decimals(self, _coin):
                return 2

            def cancel(self, _coin, _oid):
                return False

            def place_limit(self, _coin, _is_buy, _sz, _px, reduce_only=False):
                return True, "SL-1", "ok"

        strat = mod_st.Strategy(
            client=_FailCancelClient(),
            coin="HYPE",
            params=mod_st.StratParams(direction="long", stop_loss_pct=5.0),
            dry_run=False,
            replay_mode=False,
            initial_state={"qty": 1.0, "cost": 100.0, "spent": 100.0, "orders": [{"oid": "DCA-1", "role": "open", "px": 98.0, "sz": 1.0}]},
        )
        strat._save = lambda: None
        m.setattr(mod_st, "notify", lambda **_kw: None)
        assert strat._check_stop_loss(94.0) is True
        oids = [o["oid"] for o in strat.s["orders"]]
        assert "DCA-1" in oids and "SL-1" in oids


def test_ml_forecast_and_trigger_gate_and_grid_isolation(monkeypatch):
    """Verify ml_forecast/forecast.py, vol_chronos.py, MeanReversionTracker/RsiGate bool guards, and grid2/grid3 import isolation."""
    import numpy as np
    from offline.research.ml_forecast import forecast as ml_fc
    from offline.research.ml_forecast import vol_chronos as ml_vc
    from offline.research.tradeall_trigger_gate import (
        experiment_rsi_bollinger as exp8,
        experiment_rsi_dca_filter as exp9,
    )

    # 1. forecast.build_dataset handles short/non-positive arrays and returns 2D empty X
    X_empty, yd_empty, ym_empty = ml_fc.build_dataset([100.0, 0.0, -5.0])
    assert X_empty.shape == (0, len(ml_fc.FEATURES))
    assert yd_empty.shape == (0,)
    assert ym_empty.shape == (0,)

    # 2. vol_chronos.realized_vol_series handles win=0 and non-positive prices without warnings/crashes
    rv = ml_vc.realized_vol_series([100.0, 0.0, 102.0, 103.0], win=0)
    assert len(rv) == 4
    wf_empty = ml_vc.walk_forward_vol(np.full(50, np.nan), horizon=5, warmup=10, stride=5)
    assert wf_empty["n_test"] == 0

    # 3. MeanReversionTracker can be instantiated directly without prior _make_stats(tag) and rejects bool prices
    exp8.STATS.pop("direct_tag", None)
    mrt = exp8.MeanReversionTracker("direct_tag", "BTCUSDC", mode="rsi", rsi_period=2)
    assert "direct_tag" in exp8.STATS
    assert mrt.update_sign(True) == 0
    assert mrt.prev_price is None

    # 4. RsiGate rejects bool prices and hourly_ohlc filters invalid ticks
    rg = exp9.RsiGate(period=2, low=30.0)
    rg.update(True)
    assert rg.prev is None
    bars = exp9.hourly_ohlc(
        [(True, 100.0), (3600.0, False), (3600.0, -10.0), (3600.0, 105.0), (3610.0, 110.0)],
        start_ts=0.0,
        end_ts=None,
    )
    assert bars == [(105.0, 110.0, 105.0, 110.0)]


def test_glob_escape_bracket_paths_and_export_env_parsing(tmp_path):
    """Verify glob.escape handling on directory paths containing brackets and export env parsing."""
    import state_io
    import cachepaths
    import active_intents
    from orchestratorTrade import autonomous_intent_gateway as aig

    bracket_dir = tmp_path / "dir[with]brackets"
    bracket_dir.mkdir(parents=True)

    # 1. state_io.prune_rotated_archives finds and prunes archives inside bracketed directories
    base_file = bracket_dir / "state.json"
    a1 = bracket_dir / "state.json.1.archive"
    a2 = bracket_dir / "state.json.2.archive"
    a1.write_text("{}", encoding="utf-8")
    a2.write_text("{}", encoding="utf-8")
    os.utime(a1, (1000.0, 1000.0))
    os.utime(a2, (2000.0, 2000.0))
    kept = state_io.prune_rotated_archives(str(base_file), 1)
    assert kept == [str(a2)]
    assert not a1.exists() and a2.exists()

    # 2. cachepaths.migrate_legacy_filename_if_needed migrates legacy archives inside bracketed directories
    legacy_path = bracket_dir / "cache_currentprice.json"
    new_path = bracket_dir / "cache_current_price.json"
    legacy_path.write_text("{}", encoding="utf-8")
    legacy_arch = bracket_dir / "cache_currentprice.json.100.archive"
    legacy_arch.write_text("{}", encoding="utf-8")
    cachepaths.migrate_legacy_filename_if_needed(str(new_path))
    assert new_path.exists()
    assert (bracket_dir / "cache_current_price.json.100.archive").exists()

    # 3. active_intents.build_active_intent_index discovers files inside bracketed root directories
    kr_dir = bracket_dir / "kraken"
    kr_dir.mkdir()
    (kr_dir / ".state_BTCUSD.json").write_text(
        json.dumps({"orders": [{"txid": "TX-42", "side": "buy", "px": 100.0, "vol": 1.0}]}),
        encoding="utf-8",
    )
    idx = active_intents.build_active_intent_index(str(bracket_dir))
    assert len(idx["intents"]) == 1
    assert idx["intents"][0]["symbol"] == "BTCUSD"

    # 4. autonomous_intent_gateway discovers T212 configs, Kraken export pairs, and audit files inside bracketed dir
    t212_dir = bracket_dir / "212trading"
    t212_dir.mkdir()
    (t212_dir / "config.nvda.env").write_text("export T212_TICKER=NVDA_US_EQ\n", encoding="utf-8")
    eqs = aig._discover_t212_equities(str(bracket_dir))
    assert ("T212_NVDA", "NVDA_US_EQ", "NVDA_US_EQ", "USD") in eqs

    (kr_dir / "config.env").write_text('export KRAKEN_CACHE_PAIRS="SOLUSD,ATOMUSD"\n', encoding="utf-8")
    kr_pairs = aig._discover_kraken_cache_pairs(str(bracket_dir))
    assert ("SOLUSD", "SOL", "USD") in kr_pairs


def test_xstock_watch_new_coins_and_pnl_report_guards(monkeypatch):
    """Verify kraken_xstock_watch, new_coins_discovery, and verify_tools/pnl_report guards."""
    from types import SimpleNamespace
    from kraken import kraken_xstock_watch as xw
    from market_monitor import new_coins_discovery as ncd
    from verify_tools import pnl_report as pr

    # 1. kraken_xstock_watch.check_levels ignores non-positive thresholds and run_trial restores env
    sent = []
    monkeypatch.setattr(xw, "notify", lambda **kw: sent.append(kw))
    monkeypatch.setenv("XSTOCK_TP2_ALERT_PCT", "-5")
    st = {
        "allocated": {"asset": "SPCX", "qty": 10.0, "ts": 1.0},
        "pair": "SPCXUSD",
        "alerted_tp": False,
        "alerted_tp2": False,
        "alerted_sl": False,
    }
    client = SimpleNamespace(last_price=lambda _p: 100.0, balance=lambda: {"SPCX": 0})
    xw.check_levels(client, st, alloc_price=100.0, tp_pct=0.0, sl_pct=0.0, yahoo_sym="", desktop=False)
    assert sent == []

    monkeypatch.setenv("XSTOCK_TRIAL_ASSET", "SPCX")
    monkeypatch.setenv("XSTOCK_QUOTE", "USD")
    monkeypatch.setenv("XSTOCK_STATE_FILE", "/tmp/orig_xstock_state.json")
    rc = xw.run_trial(client, desktop=False)
    assert rc == 1
    assert os.environ.get("XSTOCK_STATE_FILE") == "/tmp/orig_xstock_state.json"

    # 2. NewCoinsMonitor.add_new_coin_to_watchlist respects False from price_monitor.add_symbol
    fake_factory = SimpleNamespace(
        get_all_new_coins=lambda _d: {"coinmarketcap": [{"symbol": "NEWC", "name": "NewCoin"}]},
        get_available_sources=lambda: ["coinmarketcap"],
        refresh_all=lambda: None,
    )
    reject_pm = SimpleNamespace(
        original_symbols=[],
        active_symbols=set(),
        max_monitored_symbols=10,
        add_symbol=lambda _s, preferred_source=None: False,
    )
    mon = ncd.NewCoinsMonitor(price_monitor=reject_pm, factory=fake_factory)
    assert mon.add_new_coin_to_watchlist({"symbol": "NEWC", "name": "NewCoin"}) is False
    assert mon.add_new_coin_to_watchlist({}) is False

    # 3. verify_tools.pnl_report.binance_symbol returns 0.0 unrealized when current price is unavailable
    monkeypatch.setattr(
        pr.ao,
        "get_filled_orders",
        lambda _c, _s, _st: [{"price": "100.0", "executedQty": "2.0", "side": "BUY", "time": 1000}],
    )
    monkeypatch.setattr(pr.bapi, "get_current_price", lambda _s: None)
    res = pr.binance_symbol("BTCUSDC")
    assert res["pos"] == 2.0
    assert res["unreal"] == 0.0


def test_agy_store_fix_and_verify_tools_hardening(tmp_path, monkeypatch):
    """Verify agy_store_fix bracketed glob handling and verify_tools isolation/guards."""
    import sys
    from orchestratorOS.admin import agy_store_fix as asf
    from verify_tools import check_cache_coherence as ccc
    from verify_tools import kraken_pnl_backfill as kpb
    from verify_tools import portfolio_snapshot as ps
    from verify_tools import verify_hype_facade as vhf

    # 1. check_cache_coherence can be imported without executing live checks
    assert callable(ccc.main)

    # 2. agy_store_fix.newest_daemon_log_ready finds logs in bracketed paths
    bracket_logs = tmp_path / "home[1]" / "log"
    bracket_logs.mkdir(parents=True)
    log_file = bracket_logs / "cli-20261026_120000.log"
    log_file.write_text(
        "language server process with pid 4242\nsummary store: reconciliation complete\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(asf, "LOGS", str(bracket_logs))
    assert asf.newest_daemon_log_ready(0.0, pid="4242") is True

    # 3. kraken_pnl_backfill.env_value parses export lines and closes files
    env_dir = tmp_path / "kraken_cfg"
    env_dir.mkdir()
    (env_dir / ".env").write_text('export KRAKEN_API_KEY_SPARE="spare_key_123"\n', encoding="utf-8")
    assert kpb.env_value(str(env_dir), "KRAKEN_API_KEY_SPARE") == "spare_key_123"

    # 4. portfolio_snapshot.hl_row restores sys.path and sys.modules["common"]
    before_path = list(sys.path)
    sentinel_common = object()
    monkeypatch.setitem(sys.modules, "common", sentinel_common)
    ps.hl_row()
    assert sys.path == before_path
    assert sys.modules.get("common") is sentinel_common

    # 5. verify_hype_facade._f rejects booleans and non-finite numbers
    assert vhf._f(True, -1.0) == -1.0
    assert vhf._f(float("nan"), -1.0) == -1.0
    assert vhf._f("12.5", 0.0) == 12.5

    # 6. kraken/shadow_live._print_block handles missing "current" key in configs without KeyError
    from kraken import shadow_live as k_sh
    k_sh._print_block("CUSTOM", {"bars": 10, "buyhold_pct": 1.5, "configs": {"alt": {"net_pct": 1.0, "total_pct": 1.2, "maxdd_pct": 0.5, "cycles": 1}}})

    # 7. BinancePriceStream._process_message accepts float-formatted event timestamps
    from binance_api.bapi_ws import BinancePriceStream
    ps_ws = BinancePriceStream()
    ps_ws._process_message('{"data": {"s": "BTCUSDC", "c": "75000.5", "E": "1700000000123.0"}}')
    assert ps_ws.get_price("BTCUSDC") == 75000.5


def test_provider_env_value_and_order_price_validation(tmp_path):
    from providers.base import bool_env, env_value
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.strategy_executor import ProviderError
    from providers.t212_provider import T212Provider

    # 1. providers.base.env_value strips export prefix and unquoted inline comments
    cfg_dir = tmp_path / "venue_cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.env").write_text(
        "export VENUE_FLAG=true   # inline comment\n"
        'QUOTED_HASH="abc#def"\n',
        encoding="utf-8",
    )
    assert env_value(str(cfg_dir), "VENUE_FLAG") == "true"
    assert bool_env("VENUE_FLAG", str(cfg_dir)) is True
    assert env_value(str(cfg_dir), "QUOTED_HASH") == "abc#def"

    # 2. HyperliquidProvider.round_price and submit_order/place_order reject invalid prices
    hl = HyperliquidProvider(token="HYPE", live_enabled=True)
    hl._client_tried = True
    hl._spot_pair = "@107"
    assert hl.round_price("HYPE", None) == 0.0
    assert hl.round_price("HYPE", True) == 0.0
    assert hl.round_price("HYPE", float("nan")) == 0.0
    assert hl.round_price("HYPE", -5.0) == 0.0
    assert hl.place_order("HYPE", "SELL", 0.0, 1.0) is None
    assert hl.place_order("HYPE", "SELL", True, 1.0) is None
    try:
        hl.submit_order("HYPE", "sell", 1.0, price=0.0)
        raise AssertionError("Expected ProviderError for zero limit price")
    except ProviderError:
        pass
    try:
        hl.submit_order("HYPE", "sell", 1.0, price=True)
        raise AssertionError("Expected ProviderError for bool limit price")
    except ProviderError:
        pass

    # 3. KrakenProvider.place_order and submit_order reject invalid limit prices
    kr = KrakenProvider()
    assert kr.place_order("HYPEUSD", "SELL", 0.0, 1.0) is None
    assert kr.place_order("HYPEUSD", "SELL", True, 1.0) is None
    try:
        kr.submit_order("HYPEUSD", "sell", 1.0, price=-1.0)
        raise AssertionError("Expected ProviderError for negative limit price")
    except ProviderError:
        pass

    # 4. T212Provider._signed_qty rejects non-positive quantities, and _client resolves runtime.env pacing
    try:
        T212Provider._signed_qty("BUY", -1.0)
        raise AssertionError("Expected ProviderError for negative T212 quantity")
    except ProviderError:
        pass
    (cfg_dir / "runtime.env").write_text(
        "T212_ENV=demo\nT212_MIN_GAP_SEC=0.3\nT212_PORTFOLIO_TTL_SEC=6.0\nT212_ORDER_MIN_GAP_SEC=2.1\n",
        encoding="utf-8",
    )
    (cfg_dir / ".env").write_text("T212_API_KEY=k1\nT212_API_SECRET=s1\n", encoding="utf-8")
    assert env_value(str(cfg_dir), "T212_MIN_GAP_SEC") == "0.3"
    import providers.t212_provider as t212_prov_mod
    old_t212_dir = t212_prov_mod._T212_DIR
    saved_env = {
        k: os.environ.pop(k, None)
        for k in ("T212_API_KEY", "T212_API_SECRET", "T212_ENV", "T212_MIN_GAP_SEC", "T212_PORTFOLIO_TTL_SEC", "T212_ORDER_MIN_GAP_SEC")
    }
    try:
        t212_prov_mod._T212_DIR = str(cfg_dir)
        cli = T212Provider()._client()
        assert cli.env == "demo"
        assert cli._min_gap == 0.3
        assert cli._pf_ttl == 6.0
        assert cli._order_min_gap == 2.1
    finally:
        t212_prov_mod._T212_DIR = old_t212_dir
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v


def test_venue_market_data_client_and_strategy_edge_cases(tmp_path, monkeypatch):
    """Verify null-safe Yahoo/Kraken/Hyperliquid payloads and T212/HL strategy edge cases."""
    import importlib.util
    import io
    import os
    from unittest.mock import MagicMock

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    import sys

    def _load_mod(name: str, rel_path: str):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root_dir, rel_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # 1. 212trading.market_data._chart and fetch_yahoo_candles handle null chart/indicators/quote
    t212_md = _load_mod("t212_md_test", "212trading/market_data.py")
    monkeypatch.setattr(t212_md, "http_get", lambda *a, **k: (200, b'{"chart": null}'))
    assert t212_md._chart("NVDA") == (None, [])
    monkeypatch.setattr(
        t212_md,
        "http_get",
        lambda *a, **k: (200, b'{"chart": {"result": [{"meta": null, "indicators": {"quote": [null]}}]}}'),
    )
    assert t212_md._chart("NVDA") == ({}, [])

    # 2. 212trading.analyze_volatility.fetch_ohlc handles null chart/indicators/meta
    t212_av = _load_mod("t212_av_test", "212trading/analyze_volatility.py")

    class _DummyResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(t212_av.urllib.request, "urlopen", lambda *a, **k: _DummyResp(b'{"chart": null}'))
    assert t212_av.fetch_ohlc("RGNT", "1d", "5m") == ([], {})

    # 3. kraken.market_data.fetch_ohlc_candles handles null result
    from kraken import market_data as kr_md
    monkeypatch.setattr(kr_md.urllib.request, "urlopen", lambda *a, **k: _DummyResp(b'{"error": [], "result": null}'))
    assert kr_md.fetch_ohlc_candles("HYPEUSD", 60) == []

    # 4. hyperliquid.hl_client.HLClient handles malformed/null info payloads safely
    hl_client_mod = _load_mod("hl_client_test", "hyperliquid/hl_client.py")
    HLClient = hl_client_mod.HLClient
    hl_c = HLClient.__new__(HLClient)
    hl_c.address = "0x123"
    hl_c._meta_cache = {}
    hl_c.info = MagicMock()
    hl_c.info.meta.return_value = {"universe": None}
    assert hl_c._meta() == {}
    hl_c.info.all_mids.return_value = None
    assert hl_c.mid("HYPE") is None
    assert hl_c.spot_mid("@107") is None
    hl_c.info.spot_meta.return_value = {"tokens": None, "universe": None}
    assert hl_c.resolve_spot_pair("HYPE") is None
    hl_c.info.meta_and_asset_ctxs.return_value = ({"universe": [{"name": "HYPE"}]}, None)
    assert hl_c.funding_rate("HYPE") is None

    # 5. 212trading.strategy.Strategy sanitizes invalid explicit fx_to_usd
    saved_path = list(sys.path)
    try:
        t212_strat = _load_mod("t212_strat_test", "212trading/strategy.py")
        p_t212 = t212_strat.StratParams(
            currency="USD",
            entry_amount=50.0,
            entry_discount_pct=0.2,
            dca_amount=25.0,
            dca_drop_pct=2.0,
            check_minutes=5.0,
            takeprofit_pct=1.5,
            max_budget=200.0,
            max_dca_buys=4,
            validity="DAY",
            enable_takeprofit=True,
            order_ttl_min=60.0,
            stop_loss_pct=8.0,
        )
        st_t212 = t212_strat.Strategy(
            client=None,
            ticker="NVDA_US_EQ",
            params=p_t212,
            dry_run=True,
            replay_mode=True,
            fx_to_usd=0.0,
            initial_state=t212_strat._new_state(),
        )
        assert st_t212.fx_to_usd == 1.0
    finally:
        sys.path[:] = saved_path


def test_binance_market_data_trades_intent_gateway_and_cli_edge_cases(monkeypatch, tmp_path, capsys):
    """Verify Binance market_data/trades, TrailingStop, AutonomousIntentGateway, and intelligence CLI edge cases."""
    from unittest.mock import MagicMock

    # 1. binance_api.trailing_stop passes validated float sell_fraction to TrailingCore
    from binance_api.trailing_stop import TrailingStop
    ts_bot = TrailingStop(
        api=MagicMock(),
        po=MagicMock(),
        sym=MagicMock(),
        enabled=False,
        sell_fraction="0.75",
        state_file=str(tmp_path / "ts_state.json"),
        min_profit_pct="1.5",
    )
    assert ts_bot.sell_fraction == 0.75
    assert ts_bot.core.sell_fraction == 0.75

    # 2. binance_api.market_data.fetch_klines accepts float strings for window params and rejects inconsistent OHLC bars
    from binance_api import market_data as bn_md

    class _DummyResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    raw_klines = [
        # Valid candle with float-string open/close timestamps
        ["1700000000000.0", "100.0", "105.0", "95.0", "102.0", "10.0", "1700003599000.0"],
        # Inconsistent OHLC candle (high < low) -> must be skipped
        [1700003600000, "100.0", "90.0", "95.0", "102.0", "10.0", 1700007199000],
        # Inconsistent OHLC candle (high < close) -> must be skipped
        [1700007200000, "100.0", "101.0", "95.0", "105.0", "10.0", 1700010799000],
    ]
    calls = {"n": 0}

    def _fake_urlopen(*a, **k):
        calls["n"] += 1
        return _DummyResp(json.dumps(raw_klines if calls["n"] == 1 else []).encode("utf-8"))

    monkeypatch.setattr(bn_md.urllib.request, "urlopen", _fake_urlopen)
    candles = bn_md.fetch_ohlc_candles(
        "BTCUSDC",
        interval="60",
        start_ms="1699990000000.0",
        end_ms="1700010000000.0",
        limit_per_batch="500.0",
        drop_incomplete=False,
    )
    assert len(candles) == 1
    assert candles[0]["timestamp"] == 1700000000
    assert candles[0]["close"] == 102.0

    # 3. binance_api.bapi_trades.get_my_trades_simple preserves multiple unkeyed partial fills
    from binance_api import bapi_allorders, bapi_trades
    monkeypatch.setattr(bapi_trades.sym, "validate_ordertype", lambda *a, **k: True)
    monkeypatch.setattr(bapi_trades.sym, "validate_symbols", lambda *a, **k: True)
    monkeypatch.setattr(
        bapi_allorders,
        "paginate_my_trades",
        lambda *a, **k: [
            {"symbol": "BTCUSDC", "isBuyer": True, "qty": "0.1", "price": "50000", "time": 1000},
            {"symbol": "BTCUSDC", "isBuyer": True, "qty": "0.2", "price": "50100", "time": 2000},
            "not_a_dict",
        ],
    )
    fills = bapi_trades.get_my_trades_simple("BUY", "BTCUSDC", backdays=1)
    assert len(fills) == 2

    # 4. AutonomousIntentGateway handles null history and integer intent_id without crashing
    from orchestratorTrade.autonomous_intent_gateway import _append_intent_history
    intent_obj = {"intent_id": 12345, "history": None}
    _append_intent_history(intent_obj, {"action": "TEST"})
    assert intent_obj["history"] == [{"action": "TEST"}]

    # 5. intelligence.cli handles null symbols/pillars in heartbeat and report safely
    from intelligence.cli import get_daemon_status, print_intelligence_report
    hb_file = tmp_path / "globaltelemetry_collector.heartbeat"
    hb_file.write_text(
        json.dumps({"pid": 0, "ts": 1700000000.0, "symbols": None, "intervals": None, "last_run": None}),
        encoding="utf-8",
    )
    st = get_daemon_status(str(tmp_path))
    assert st["symbols"] == []
    assert st["intervals"] == {}
    assert st["last_run"] == {}

    print_intelligence_report({
        "symbol": "BTCUSDC",
        "pillar1_internal": None,
        "pillar2_external": None,
        "pillar3_sentiment": None,
        "pillar4_macro": None,
        "guard_preview": None,
    })
    out = capsys.readouterr().out
    assert "MARKET INTELLIGENCE AUDIT — BTCUSDC" in out

    # 6. HyperliquidProvider, KrakenProvider, and T212Provider guard against null/non-dict nested payloads
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.t212_provider import T212Provider

    hl_prov = HyperliquidProvider()
    hl_mock = MagicMock()
    hl_mock.info.spot_user_state.return_value = {"balances": None}
    hl_prov._client = hl_mock
    hl_prov._client_tried = True
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xabc")
    assert hl_prov.free_balance("USDC") is None

    kr_cli = MagicMock()
    kr_cli.open_orders.return_value = {"O1": "not_a_dict"}
    kr_cli.closed_orders.return_value = {"O2": {"cl_ord_id": "abc", "descr": None}}
    kr_prov = KrakenProvider(client=kr_cli)
    assert kr_prov.order_by_client_id("HYPEUSD", "abc") is None

    t212_cli = MagicMock()
    t212_cli.get_portfolio.return_value = ["not_a_dict", {"ticker": "NVDA_US_EQ", "quantity": 2.0}]
    t212_prov = T212Provider(client=t212_cli)
    assert t212_prov.free_balance("NVDA_US_EQ") == 2.0

    # 7. PositionExitManager, PairCoordinator.from_state, and PriceChecker edge cases
    import math
    from strategies.position_exit_manager import (
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
        ReentrySnapshot,
    )
    from strategies.rtrade_pair import PairCoordinator
    from market_monitor.pricechecker import PriceChecker

    snap = PositionSnapshot(symbol="BTCUSDC", current_price="110.0", avg_buy_price="100.0")
    assert snap.is_valid_price is True
    assert math.isclose(snap.gain_ratio, 0.10)

    rsnap = ReentrySnapshot(symbol="BTCUSDC", current_price="95.0", last_sell_price="100.0")
    assert rsnap.is_valid_price is True
    assert math.isclose(rsnap.sale_drop_ratio, 0.05)

    pem = PositionExitManager(config=PositionExitConfig(hard_tp_enabled=True, hard_tp_pct=0.05, hard_tp_frac=0.5))
    corrupted_store = {
        "sl_bounce_state": None,
        "sl_episode_state": "invalid",
        "tp_ride_state": {"BTCUSDC": "not_a_dict"},
        "reentry_state": {"BTCUSDC": None},
    }
    pem.bind_state(corrupted_store)
    assert isinstance(pem.sl_bounce_state, dict)
    assert "BTCUSDC" not in pem.tp_ride_state
    pem.record_hard_tp("BTCUSDC", timestamp="invalid")
    pem.hard_tp_last["BTCUSDC"] = "not_a_float"
    dec = pem.evaluate_take_profit(snap)
    assert dec.should_exit is True

    with pytest.raises(ValueError, match="tickets"):
        PairCoordinator.from_state(
            MagicMock(),
            MagicMock(step_interval_sec=1.0),
            {"pair_id": "p1", "qty": 1.0, "start_side": "BUY", "phase": "paired", "tickets": ["not_a_dict"]},
        )

    checker_cache = MagicMock()
    checker_cache.symbol_added_time = None
    checker = PriceChecker(checker_cache, config={"default": None, "dynamic": "bad"})
    assert isinstance(checker.config["default"], dict)
    assert checker._is_dynamic_symbol("BTC") is False

    # 8. T212 order_manager/strategy, Kraken cachemanager/trailing_stop, and Binance trailing_stop state resilience
    import importlib.util
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t212_dir = os.path.join(repo_root, "212trading")
    if t212_dir not in sys.path:
        sys.path.insert(0, t212_dir)
    t212_strat_spec = importlib.util.spec_from_file_location(
        "t212_strategy_test_mod", os.path.join(t212_dir, "strategy.py")
    )
    t212_strat_mod = importlib.util.module_from_spec(t212_strat_spec)
    sys.modules["t212_strategy_test_mod"] = t212_strat_mod
    try:
        t212_strat_spec.loader.exec_module(t212_strat_mod)
    finally:
        sys.modules.pop("t212_strategy_test_mod", None)
        if t212_dir in sys.path:
            sys.path.remove(t212_dir)
    sanitized_t212 = t212_strat_mod._sanitize_t212_state({
        "qty": "2.5",
        "cost_usd": "250.0",
        "orders": ["not_a_dict", {"id": "O1", "side": "SELL", "qty": 1.0, "limit": 120.0}],
        "tp_sold_levels": [None, "2.0", "bad"],
        "tp_fail_until": None,
        "buy_backoff_until": None,
        "sl_pending_exit": "corrupted",
        "tp_pending_exit": 123,
    })
    assert sanitized_t212["qty"] == 2.5
    assert len(sanitized_t212["orders"]) == 1
    assert sanitized_t212["tp_sold_levels"] == [2.0]
    assert "tp_fail_until" not in sanitized_t212
    assert "buy_backoff_until" not in sanitized_t212
    assert sanitized_t212["sl_pending_exit"] is None
    assert sanitized_t212["tp_pending_exit"] is None

    from kraken.kraken_cachemanager import _load_initial_cache
    bad_kraken_cache = tmp_path / "cache_trade_kraken_bad.json"
    bad_kraken_cache.write_text(
        json.dumps({
            "items": {
                "HYPEUSD": [
                    "not_a_dict",
                    {"id": None, "time": 1000},
                    {"id": "T2", "time": "2000", "price": 10.0, "qty": 1.0},
                    {"id": "T1", "time": 1000, "price": 9.5, "qty": 1.0},
                ]
            },
            "fetchtime": {"HYPEUSD": 2000},
        }),
        encoding="utf-8",
    )
    loaded_kc = _load_initial_cache(str(bad_kraken_cache))
    assert [r["id"] for r in loaded_kc["items"]["HYPEUSD"]] == ["T1", "T2"]

    # 9. Volatility, KalmanTrigger, SmartStopLossGuard, NewsFeedCollector, and GeopoliticalThreatAnalyzer resilience
    from intelligence.internal.state.volatility import (
        RollingVolatilityTracker,
        calculate_volatility_1h,
        vol_pct_to_ratio,
        vol_ratio_to_pct,
    )
    import numpy as np
    noisy_prices = [None, True] + [100.0 + (i % 3) * 0.5 for i in range(25)] + ["bad", False]
    v1h = calculate_volatility_1h(noisy_prices, 60.0)
    assert v1h is not None and v1h > 0.0
    np_prices = np.asarray([100.0 + (i % 3) * 0.5 for i in range(25)], dtype=float)
    assert calculate_volatility_1h(np_prices, 60.0) == v1h
    assert calculate_volatility_1h(np.asarray([True] * 25, dtype=bool), 60.0) is None
    assert vol_pct_to_ratio(True) == 0.0
    assert vol_pct_to_ratio(None) == 0.0
    assert vol_ratio_to_pct(True) == 0.0
    assert vol_ratio_to_pct("0.025") == 2.5
    rvt = RollingVolatilityTracker(window_size="5")
    assert rvt.window_size == 20

    from intelligence.internal.triggers.kalman_trigger import KalmanTrend, KalmanTrendTrigger
    kt = KalmanTrend()
    kt.x = [100.0, 0.0]
    kt.last_ts = None
    out_kt = kt.update("1000.0", "105.0", 0.01)
    assert out_kt["trend"] == 0 and kt.last_ts == 1000.0
    ktt = KalmanTrendTrigger()
    ktt.filter.update(1000.0, 100.0, 0.001)
    _, ev_kt = ktt.evaluate("BTCUSDC", "1060.0", "110.0", 0.001)
    if ev_kt is not None:
        assert isinstance(ev_kt.price, float) and isinstance(ev_kt.ts, float)

    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard
    sl_guard = SmartStopLossGuard(
        mode=None,
        max_threshold_pct="bad",
        grace_seconds=True,
        disaster_buffer_pct=-5.0,
        min_whale_long_pct=2.5,
        min_taker_ratio=-1.0,
    )
    assert sl_guard.mode == "enforce"
    assert sl_guard.max_threshold_pct == 6.0
    assert sl_guard.grace_seconds == 120.0
    assert sl_guard.disaster_buffer_pct == 0.85
    assert sl_guard.min_whale_long_pct == 0.60
    assert sl_guard.min_taker_ratio == 0.85

    from intelligence.macro.news_feed_collector import NewsFeedCollector
    nfc = NewsFeedCollector(history_file=str(tmp_path / "news_hist.json"))
    assert nfc._normalize_title(None) == ""
    nfc._aggregated_pool["bad1"] = "not_a_tuple"
    nfc._aggregated_pool["bad2"] = (True, "not_headline")
    snap_nf = nfc.get_pool_snapshot(now_ts=2000.0)
    assert snap_nf.total_fetched == 0

    # 10. Providers (Kraken ohlc_series, base aggregate_fills_by_order_id, quantity, replay_provider, _parse_tranches)
    from providers.kraken_provider import KrakenProvider
    from providers.strategy_executor import ProviderError
    kp = KrakenProvider()
    with pytest.raises(ProviderError):
        kp.ohlc_series("HYPEUSD", True)
    with pytest.raises(ProviderError):
        kp.ohlc_series("HYPEUSD", 0)

    from providers.base import aggregate_fills_by_order_id
    agg_rows = [
        {"orderId": "O10", "side": "BUY", "price": 10.0, "qty": 2.0, "timestamp": 1000},
        {"orderId": "O11", "side": "SELL", "price": 12.0, "qty": 1.0, "timestamp": 2000},
    ]
    assert len(aggregate_fills_by_order_id(agg_rows, side="b")) == 1
    assert len(aggregate_fills_by_order_id(agg_rows, side="s")) == 1

    from providers.quantity import balance_cap_quantity, resolve_assets
    assert resolve_assets("  btcusdc  ") == ("BTC", "USDC")
    cap_b, asset_b = balance_cap_quantity(lambda _a: 100.0, "BTCUSDC", " b ", 50.0)
    assert cap_b == 2.0 and asset_b == "USDC"
    cap_s, asset_s = balance_cap_quantity(lambda _a: 3.5, "BTCUSDC", " s ", 50.0)
    assert cap_s == 3.5 and asset_s == "BTC"

    from providers.replay_provider import ReplayMarketDataProvider
    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0), (1060.0, 51000.0)]})
    rp.advance("BTCUSDC")
    assert rp.place_order("BTCUSDC", "b", 50000.0, 1.0) is not None
    assert len(rp.get_orders("BTCUSDC", "b", 3600.0)) == 1
    assert rp.place_order("BTCUSDC", "s", 50000.0, 0.5) is not None
    assert len(rp.get_orders("BTCUSDC", "s", 3600.0)) == 1

    from strategies.spot_engine import _parse_tranches
    assert _parse_tranches(None) == []

    # 11. Binance API & Hyperliquid helper guards (order_type normalization, _last_opposite_fill_price_api, HL cancel/market_data/signals)
    import importlib.util
    from types import SimpleNamespace
    import time
    from binance_api import bapi, bapi_placeorder

    def _load_hl_mod(mod_name: str, filename: str):
        hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
        orig_path = list(sys.path)
        try:
            if hl_dir not in sys.path:
                sys.path.insert(0, hl_dir)
            spec = importlib.util.spec_from_file_location(mod_name, os.path.join(hl_dir, filename))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            return mod
        finally:
            sys.modules.pop(mod_name, None)
            sys.path[:] = orig_path

    hl_client = _load_hl_mod("_test_hl_client", "hl_client.py")
    hl_md = _load_hl_mod("_test_hl_md", "market_data.py")
    hl_signals = _load_hl_mod("_test_hl_signals", "signals.py")

    with pytest.raises(ValueError):
        bapi.get_open_orders(True, "BTCUSDC")
    with pytest.raises(ValueError):
        bapi_placeorder.order_candidate_price(True, 50000.0, 50000.0)
    assert bapi_placeorder.order_candidate_price(" buy ", 50000.0, 49000.0) == 49000.0

    monkeypatch.setattr(
        bapi_placeorder.client,
        "get_my_trades",
        lambda symbol, limit=200: [
            {"isBuyer": False, "price": "51000.0"},
            {"isBuyer": False, "price": "0.0"},
        ],
    )
    assert bapi_placeorder._last_opposite_fill_price_api("BTCUSDC", " buy ") == 51000.0
    monkeypatch.setattr(bapi_placeorder.client, "get_my_trades", lambda symbol, limit=200: None)
    assert bapi_placeorder._last_opposite_fill_price_api("BTCUSDC", "BUY") is None

    dummy_hl = object.__new__(hl_client.HLClient)
    captured_cancel = []
    dummy_hl.exchange = SimpleNamespace(
        cancel=lambda coin, oid: (captured_cancel.append((coin, oid)) or {"status": "ok", "response": {"data": {"statuses": ["success"]}}})
    )
    assert dummy_hl.cancel("HYPE", "42") is True
    assert captured_cancel == [("HYPE", 42)]
    assert dummy_hl.cancel("HYPE", "-5") is False
    assert dummy_hl.cancel("HYPE", "not_an_oid") is False

    monkeypatch.setattr(hl_md, "_post_info", lambda body, timeout=25: [])
    assert hl_md.fetch_ohlc_candles("HYPE", " 60 ", lookback_hours=2) == []
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", "1h", start_ms=2000, end_ms=1000)
    with pytest.raises(ValueError):
        hl_md.fetch_ohlc_candles("HYPE", "1h", timeout=0)

    sig_file = tmp_path / "hl_signal_ms.json"
    sig_file.write_text(
        json.dumps({"trend": " UP ", "confidence": 0.8, "ts": int(time.time() * 1000)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SIGNAL_MAX_AGE_MIN", "15")
    sig_res = hl_signals._file_signal(str(sig_file))
    assert sig_res["trend"] == "up"
    assert sig_res["confidence"] == 0.8
    assert sig_res["source"] == "file"

    # 12. Hyperliquid replay/strategy and 212trading strategy/market_data guards
    hl_strat = _load_hl_mod("_test_hl_strat_p", "strategy.py")
    sp_hl = hl_strat.StratParams.from_dict({"enable_takeprofit": "false", "signal_gate": "0"})
    assert sp_hl.enable_takeprofit is False
    assert sp_hl.signal_gate is False

    hl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperliquid")
    orig_path = list(sys.path)
    try:
        if hl_dir not in sys.path:
            sys.path.insert(0, hl_dir)
        sys.modules["strategy"] = hl_strat
        spec_rp = importlib.util.spec_from_file_location("_test_hl_replay", os.path.join(hl_dir, "replay.py"))
        hl_replay = importlib.util.module_from_spec(spec_rp)
        sys.modules["_test_hl_replay"] = hl_replay
        spec_rp.loader.exec_module(hl_replay)
    finally:
        sys.modules.pop("_test_hl_replay", None)
        sys.modules.pop("strategy", None)
        sys.path[:] = orig_path

    empty_rp = hl_replay.run_replay([], params={"budget": None})
    assert empty_rp["total"] == 0.0
    assert empty_rp["initial_capital"] == 500.0

    t212_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "212trading")
    orig_path = list(sys.path)
    try:
        if t212_dir not in sys.path:
            sys.path.insert(0, t212_dir)
        spec_t212_md = importlib.util.spec_from_file_location("_test_t212_md2", os.path.join(t212_dir, "market_data.py"))
        t212_md2 = importlib.util.module_from_spec(spec_t212_md)
        sys.modules["_test_t212_md2"] = t212_md2
        spec_t212_md.loader.exec_module(t212_md2)
        sys.modules["market_data"] = t212_md2
        spec_t212_st = importlib.util.spec_from_file_location("_test_t212_st2", os.path.join(t212_dir, "strategy.py"))
        t212_st2 = importlib.util.module_from_spec(spec_t212_st)
        sys.modules["_test_t212_st2"] = t212_st2
        spec_t212_st.loader.exec_module(t212_st2)
    finally:
        sys.modules.pop("_test_t212_md2", None)
        sys.modules.pop("_test_t212_st2", None)
        sys.modules.pop("market_data", None)
        sys.path[:] = orig_path

    from offline.research.hybrid_reentry import run_equities
    base_t212_cfg = dict(run_equities.BASE_CONFIG, YAHOO_SYMBOL="NVDA")
    assert t212_md2.parse_interval_minutes(" 1h ") == 60
    with pytest.raises(ValueError):
        t212_md2.fetch_yahoo_candles("NVDA", "1mo", "1h", timeout=0)
    with pytest.raises(ValueError, match="STRAT_MAX_DCA_BUYS"):
        t212_st2.StratParams.from_env(dict(base_t212_cfg, STRAT_MAX_DCA_BUYS="3.5"))
    with pytest.raises(ValueError, match="step"):
        t212_st2.generate_tp_ladder(1.0, 5.0, steps="2.5")
    monkeypatch.setattr(t212_st2, "get_eur_usd", lambda: None)
    assert t212_st2.Strategy._fx_to_usd(None, " eur ") == 1.08

    # 13. KrakenClient null open/closed order maps and T212Client non-list/non-dict payload guards
    from kraken.kraken_client import KrakenClient
    kc = KrakenClient("k", "c2VjcmV0")
    monkeypatch.setattr(kc, "_private", lambda method, data=None, fresh=False: {"open": None, "closed": None})
    assert kc.open_orders() == {}
    assert kc.closed_orders() == {}
    monkeypatch.setattr(kc, "_public", lambda method, params=None, fresh=False: {"last": 123, "HYPEUSD": {"pair_decimals": 3}})
    assert kc.pair_info("HYPEUSD") == {"pair_decimals": 3}

    orig_path = list(sys.path)
    try:
        if t212_dir not in sys.path:
            sys.path.insert(0, t212_dir)
        spec_tc = importlib.util.spec_from_file_location("_test_t212_cli2", os.path.join(t212_dir, "t212_client.py"))
        t212_cli_mod = importlib.util.module_from_spec(spec_tc)
        sys.modules["_test_t212_cli2"] = t212_cli_mod
        spec_tc.loader.exec_module(t212_cli_mod)
        spec_tb = importlib.util.spec_from_file_location("_test_t212_bot2", os.path.join(t212_dir, "t212_bot.py"))
        t212_bot_mod = importlib.util.module_from_spec(spec_tb)
        sys.modules["_test_t212_bot2"] = t212_bot_mod
        spec_tb.loader.exec_module(t212_bot_mod)
    finally:
        sys.modules.pop("_test_t212_cli2", None)
        sys.modules.pop("_test_t212_bot2", None)
        sys.path[:] = orig_path

    tc = t212_cli_mod.T212Client("k", "s", env="demo", min_gap_sec=0.0, portfolio_ttl_sec=1.0, order_min_gap_sec=0.0)
    monkeypatch.setattr(t212_cli_mod, "http_get", lambda url, headers=None: (200, b'{"not": "a_list"}'))
    assert tc.list_instruments() is None
    monkeypatch.setattr(t212_cli_mod, "http_get", lambda url, headers=None: (200, b'["bad", {"ticker": "NVDA_US_EQ", "isin": ""}]'))
    assert tc.list_instruments() == [{"ticker": "NVDA_US_EQ", "isin": ""}]
    assert t212_bot_mod.verify_isin(tc, " nvda_us_eq ", "US67066G1040") is True
    monkeypatch.setattr(t212_cli_mod, "http_post_json", lambda url, payload=None, headers=None: (200, b'["unexpected_list"]'))
    st_code, payload_dict = tc._place_order("limit", {"ticker": "NVDA_US_EQ"})
    assert st_code == 200 and isinstance(payload_dict, dict)

    # 14. discovery.news, shadow_runtime, instrument_registry, orchestratorOS.agy.paths, and watchdog_common guards
    import pandas as pd
    from discovery import news as d_news
    from instrument_registry import InstrumentSpec, select_instruments
    from orchestratorOS.agy import paths as agy_paths
    from orchestratorOS.lib import watchdog_common as wd_common
    from shadow_runtime import require_shadow_interval

    df_cmc_sample = pd.DataFrame([
        {"name": "Bitcoin", "symbol": "BTC", "launch_date": pd.Timestamp("2020-01-01"), "price": 60000.0, "change_24h": 1.0, "change_7d": 2.0, "website_slug": "btc"},
        {"name": "BadBool", "symbol": "BAD_BOOL", "launch_date": pd.Timestamp("2024-01-01"), "price": True, "change_24h": 0.0, "change_7d": 0.0, "website_slug": "bb"},
        {"name": "BadNan", "symbol": "BAD_NAN", "launch_date": pd.Timestamp("2024-01-02"), "price": float("nan"), "change_24h": 0.0, "change_7d": 0.0, "website_slug": "bn"},
        {"name": "Ethereum", "symbol": "ETH", "launch_date": pd.Timestamp("2022-01-01"), "price": 3000.0, "change_24h": 3.0, "change_7d": 4.0, "website_slug": "eth"},
    ])
    df_new, *_ = d_news.find_common_coins_and_sort(
        1,
        df_cmc_sample,
        {"BTC": 60010.0, "BAD_BOOL": 1.0, "BAD_NAN": 1.0, "ETH": 3005.0},
    )
    assert list(df_new["symbol"]) == ["ETH"]
    df_new_bool, *_ = d_news.find_common_coins_and_sort(True, df_cmc_sample, {"BTC": 60010.0})
    assert list(df_new_bool["symbol"]) == ["BTC"]

    with pytest.raises(ValueError):
        require_shadow_interval(1, True, "TEST_BOT")

    spec_inst = InstrumentSpec(
        name="BINANCE_BTC",
        provider="binance",
        symbol="BTCUSDC",
        base="BTC",
        quote="USDC",
        enabled=True,
        isolation="own_ledger",
        market_hours="24x7",
        params={"role.mt": "yes", "role.rtrade": "yes", "gain": "3.5"},
    )
    assert spec_inst.has_role("  MT ") is True
    with pytest.raises(ValueError):
        spec_inst.has_role(True)
    assert spec_inst.setting(" GAIN ") == "3.5"
    with pytest.raises(ValueError):
        spec_inst.setting(True)
    assert len(select_instruments(role=" MT ", provider=" BINANCE ")) >= 1
    with pytest.raises(ValueError):
        select_instruments(role=True)
    with pytest.raises(ValueError):
        select_instruments(provider=True)

    monkeypatch.setenv("AGY_BOT_HOME", str(tmp_path / "bot_home"))
    monkeypatch.setenv("AGY_UI_CLI_BASE", str(tmp_path / "ui_cli"))
    monkeypatch.setenv("AGY_BOT_BIN", str(tmp_path / "agy_bot_bin"))
    monkeypatch.setenv("AGY_FIX_HOME", "   ")
    assert agy_paths.resolve_bot_cli_base("   ").startswith(str(tmp_path / "bot_home"))
    assert agy_paths.resolve_ui_cli_base("   ") == str(tmp_path / "ui_cli")
    assert agy_paths.resolve_agy_bot_bin("   ") == str(tmp_path / "agy_bot_bin")
    assert agy_paths.resolve_store_lock_path("   ").endswith(".agy_store_fix.lock")

    sent_urls = []
    monkeypatch.setenv("NTFY_TOPIC_ERROR", "   ")
    monkeypatch.setenv("PHONE_ALERT_URL", " https://ntfy.sh/fallback_topic ")
    monkeypatch.setattr(
        wd_common.AlertNotifier,
        "send_phone_webhook_batch",
        lambda events, webhook_url=None: (sent_urls.append(webhook_url) or True),
    )
    assert wd_common.send_ntfy("Test", "Body") is True
    assert sent_urls == ["https://ntfy.sh/fallback_topic"]

    # 15. persistence, survival, mean_reversion_trigger, trend_significance_guard, exhaustion_guard, parabolic_guard, and pricefetcher guards
    import numpy as np
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.internal.state.persistence import (
        _extract_finite_prices,
        calculate_mann_kendall,
        classify_hurst_regime,
    )
    from intelligence.internal.state.survival import block_slopes, hybrid_T
    from intelligence.internal.triggers.mean_reversion_trigger import MeanReversionTrigger
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
    from market_monitor.pricefetcher import StockYahooPricePlatform

    assert list(_extract_finite_prices([True, 100.0, False, 102.0])) == [100.0, 102.0]
    assert len(_extract_finite_prices(np.array([True, False]))) == 0
    assert calculate_mann_kendall([True, False, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]) == (0, 0.0, 1.0)
    assert classify_hurst_regime(0.60, lo="bad", hi=True) == "persistent"
    assert classify_hurst_regime(0.50, lo=0.60, hi=0.40) == "random_walk"

    bt_arr, bs_arr = block_slopes(np.arange(10, dtype=float) * 3600.0, np.arange(8, dtype=float) + 100.0, window_h=True, step_h="bad")
    assert len(bt_arr) == 0 and len(bs_arr) == 0
    ht_bool = hybrid_T([True, False], prior_T="12.0", t_min=30, t_max=4)
    assert ht_bool["n"] == 0 and ht_bool["T"] == 12
    assert hybrid_T(True)["n"] == 0

    mrt = MeanReversionTrigger(mode=" RSI ", bb_k=-1.0)
    assert mrt.mode == "rsi" and mrt.bb_k == 2.0

    tsg = TrendSignificanceGuard(max_p_value=0.05)
    entry_trig = TriggerEvent(
        action=TriggerAction.ENTRY,
        side=TriggerSide.BUY,
        source="test",
        symbol="BTCUSDC",
        price=100.0,
        strength=1.0,
    )
    dec_sparse = tsg.evaluate(entry_trig, [None, True, False, float("nan"), 100.0, 101.0, 102.0, 103.0, 104.0])
    assert dec_sparse.allowed is True and dec_sparse.reason == "insufficient_history_for_mk_test"

    weg = WeibullExhaustionGuard(policy=" OFF ", exhausted_scale=2.5)
    assert weg.policy == "off" and weg.exhausted_scale == 1.0
    weg_enforced = WeibullExhaustionGuard(policy=" VETO ")
    assert weg_enforced.check("BTCUSDC", True, 86400.0 * 10).allowed is True

    psg = ParabolicSurgeGuard()
    assert psg.check("BTCUSDC", True, 100.0).allowed is True
    assert psg.check("BTCUSDC", " buy ", 100.0, price_history=[(True, 99.0)]).allowed is True

    yp = StockYahooPricePlatform(tickers=["NVDA_US_EQ"])

    class _DummyYahooResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"chart": None}

    monkeypatch.setattr("market_monitor.pricefetcher.requests.get", lambda *a, **k: _DummyYahooResp())
    assert yp.get_price("NVDA_US_EQ") is None

    # 16. External/Sentiment/Macro collectors, triggers, and guards whitespace/bool/bounds hardening
    from intelligence.external.collectors.bybit_liquidations import BybitLiquidationCollector, LiquidationSummary
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetry, DerivativesTelemetryCollector
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector, OrderbookSnapshot
    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector, WhalePositioningSnapshot
    from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
    from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
    from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAssessment
    from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
    from intelligence.sentiment.guards.extreme_greed_guard import ExtremeGreedGuard
    from intelligence.sentiment.guards.high_stake_guard import compute_24h_price_metrics, extract_symbol_recent_fills
    from intelligence.sentiment.guards.panic_washout_guard import PanicWashoutGuard

    blc = BybitLiquidationCollector(symbols=["  btcusdt ", "", None])
    assert blc._symbols == ["BTCUSDT"]

    assert DerivativesTelemetry.from_dict(None).symbol == ""
    assert DerivativesTelemetryCollector(cache_dir=None).fetch("   ", allow_network=False) is None
    assert OrderbookSnapshot.from_dict("bad").symbol == ""
    assert OrderbookDepthCollector(cache_dir=None).fetch("   ", allow_network=False) is None
    assert WhalePositioningCollector(cache_dir=None).fetch("   ", allow_network=False) is None

    lcg = LiquidationCascadeGuard(max_active_cascade_usd=100_000.0)
    liq_sum = LiquidationSummary(
        symbol="BTCUSDT",
        window_sec=60.0,
        long_liq_usd=250_000.0,
        short_liq_usd=0.0,
        net_usd=-250_000.0,
        capitulation_ratio=1.0,
        squeeze_ratio=0.0,
        events_count=5,
        last_event_ts=1000.0,
    )
    assert lcg.check("BTCUSDT", " buy ", summary=liq_sum).allowed is False
    assert lcg.check("BTCUSDT", True, summary=liq_sum).allowed is True

    wdg = WhaleDivergenceGuard(
        block_short_covering_fakeout="false",
        block_aggressive_shorting="0",
        block_liquidation_cascade="no",
    )
    assert wdg.block_short_covering_fakeout is False
    assert wdg.block_aggressive_shorting is False
    assert wdg.block_liquidation_cascade is False
    wdg_strict = WhaleDivergenceGuard()
    whale_snap = WhalePositioningSnapshot(
        symbol="BTCUSDC",
        top_traders_long_ratio=1.2,
        top_traders_long_pct=0.55,
        taker_buy_sell_ratio=1.0,
        taker_buy_vol_usd=100.0,
        taker_sell_vol_usd=100.0,
        open_interest_usd=1000.0,
        open_interest_1h_change_pct=-1.0,
        divergence_regime=" SHORT_COVERING ",
        ts=1000.0,
    )
    assert wdg_strict.check("BTCUSDC", " buy ", snapshot=whale_snap).allowed is False

    lct = LiquidationCapitulationTrigger(ratio_threshold=75.0)
    assert lct.ratio_threshold == 0.75
    wat = WhaleAccumulationTrigger(min_top_traders_long_pct=60.0, min_top_traders_short_pct=60.0)
    assert wat.min_top_traders_long_pct == 0.60 and wat.min_top_traders_short_pct == 0.60

    egg = ExtremeGreedGuard(hard_veto_threshold=90)
    fg_greed = FearGreedSnapshot(
        value=95,
        classification="Extreme Greed",
        timestamp=1000.0,
        historical_values=(95,),
        trend_7d_change=10,
        is_extreme_fear=False,
        is_extreme_greed=True,
    )
    assert egg.check("BTCUSDC", " buy ", fg_greed).allowed is False

    pwg = PanicWashoutGuard(severe_panic_fng_threshold=15)
    fg_panic = FearGreedSnapshot(
        value=10,
        classification="Extreme Fear",
        timestamp=1000.0,
        historical_values=(10, 25),
        trend_7d_change=-15,
        is_extreme_fear=True,
        is_extreme_greed=False,
    )
    assert pwg.check("BTCUSDC", " buy ", fear_greed_snapshot=fg_panic).allowed is False

    gsg = GeopoliticalShockGuard()
    geo_shock = SimpleNamespace(
        threat_level=" CRITICAL_SHOCK ",
        risk_score=0.95,
        recommended_brake=" HARD_VETO_NEW_BUYS ",
        summary="crisis",
        ts=1000.0,
    )
    assert gsg.check("BTCUSDC", " buy ", geo_shock).allowed is False

    m24_unsorted = compute_24h_price_metrics([[2000.0, 110.0], [1000.0, 100.0]])
    assert m24_unsorted.get("chg_15m_pct") == 10.0
    fills_ws = extract_symbol_recent_fills([{"side": " buy ", "price": "100.0", "qty": "2.0", "timestamp": 1000}])
    assert fills_ws is not None and fills_ws["active_avg_buy_px"] == 100.0


def test_composite_and_macro_advisor_robustness(tmp_path):
    """Verify composite market intelligence, geopolitical analyzer, sentiment advisor, and post-stop guard normalization."""
    import json
    from types import SimpleNamespace
    from intelligence.composite import CompositeMarketIntelligence
    from intelligence.daemon import IntelligenceTelemetryDaemon
    from intelligence.globaltelemetry_collector import GlobalTelemetryCollector
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor
    from intelligence.sentiment.guards.post_stop_reentry_guard import (
        PostStopReentryGuard,
        get_post_stop_reentry_guard,
    )

    cmi = CompositeMarketIntelligence()
    # Non-sequence prices_for_vol and string sample_rate_sec must not raise TypeError
    eval_res = cmi.evaluate(
        "BTCUSDC",
        100.0,
        ts=1000.0,
        prices_for_vol=True,  # type: ignore[arg-type]
        sample_rate_sec="60",  # type: ignore[arg-type]
    )
    assert eval_res.symbol == "BTCUSDC"

    mid = IntelligenceTelemetryDaemon(
        symbols=["BTCUSDC"],
        cache_dir=str(tmp_path),
        skip_macro="false",  # type: ignore[arg-type]
        skip_external="0",  # type: ignore[arg-type]
    )
    assert mid.skip_macro is False and mid.skip_external is False

    gtcd = GlobalTelemetryCollector(
        symbols=["BTCUSDC"],
        cache_dir=str(tmp_path),
        skip_news="false",  # type: ignore[arg-type]
        skip_external="0",  # type: ignore[arg-type]
    )
    assert gtcd.skip_news is False and gtcd.skip_external is False

    geo_state = tmp_path / "geopolitical_threat_eval.json"
    geo_state.write_text(
        json.dumps(
            {
                "threat_level": " elevated ",
                "risk_score": "0.65",
                "summary": "cached summary",
                "recommended_brake": " downscale_50 ",
                "headlines_analyzed": "5",
                "ts": 1000.0,
                "is_fallback": "false",
            }
        ),
        encoding="utf-8",
    )
    gta = GeopoliticalThreatAnalyzer(state_file=str(geo_state))
    assert gta._cached_assessment is not None
    assert gta._cached_assessment.threat_level == "ELEVATED"
    assert gta._cached_assessment.recommended_brake == "DOWNSCALE_50"
    assert gta._cached_assessment.is_fallback is False

    sa_state = tmp_path / "sentiment_advisor_eval.json"
    sa_state.write_text(
        json.dumps(
            {
                "market_bias": " bullish ",
                "risk_level": " low ",
                "confidence": "0.8",
                "summary": "ok",
                "key_risks": ["r1"],
                "recommended_action": " accumulate ",
                "ts": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    sa = SentimentAdvisor(cache_file=str(sa_state))
    assert sa._cached_assessment is not None
    assert sa._cached_assessment.market_bias == "BULLISH"
    assert sa._cached_assessment.risk_level == "LOW"
    assert sa._cached_assessment.recommended_action == "ACCUMULATE"

    psg = PostStopReentryGuard(notify_enabled="false")  # type: ignore[arg-type]
    assert psg.notify_enabled is False
    psg_cfg = get_post_stop_reentry_guard(
        margins={
            "post_stop_reentry_mode": "shadow",
            "post_stop_reentry_timeout_sec": "0",
            "post_stop_reentry_fallback": "allow",
            "post_stop_reentry_notify": "false",
            "post_stop_reentry_cache_ttl_sec": "600",
        },
        gemini_client=SimpleNamespace(_custom_runner=None),
    )
    assert psg_cfg.mode == "shadow" and psg_cfg.notify_enabled is False

    from market_monitor.new_coins_discovery import is_trackable_symbol
    from market_monitor.pricechecker import PriceAlert, PriceChecker

    assert is_trackable_symbol("usdc") is False
    assert is_trackable_symbol(" btc ") is False
    assert is_trackable_symbol("NEWCOIN") is True

    pa = PriceAlert(" BTCUSDC ", " UP ", "100.5", "95.0", "5.78", "4.1")  # type: ignore[arg-type]
    assert pa.symbol == "BTCUSDC" and pa.alert_type == "up" and pa.current_price == 100.5
    assert "$100.5000" in str(pa)

    pc = PriceChecker(
        SimpleNamespace(symbol_added_time={}),
        config={"per_coin": {"BTC": {"up_percent": 9.5, "down_percent": 11.5}}},
    )
    th = pc._get_thresholds_for_symbol(" btcusdc ")
    assert th["up_percent"] == 9.5 and th["down_percent"] == 11.5


def test_intelligence_cli_bapi_ws_and_pricefetcher_robustness(tmp_path):
    """Verify intelligence.cli, bapi_ws_health, bapi_ws, and pricefetcher handle string booleans, whitespace symbols, and FearGreedSnapshot schema."""
    import json
    from binance_api.bapi_ws import BinancePriceStream
    from binance_api.bapi_ws_health import mark_ws_available, should_poll_for_manager
    from intelligence.cli import inspect_symbol_intelligence
    from market_monitor.pricefetcher import StockYahooPricePlatform

    # 1. intelligence.cli: whitespace symbol + FearGreedSnapshot timestamp/classification fields
    fg_file = tmp_path / "fear_greed_collect.json"
    fg_file.write_text(
        json.dumps({"value": 24, "classification": "Extreme Fear", "timestamp": 1700000000.0}),
        encoding="utf-8",
    )
    info = inspect_symbol_intelligence(" btcusdc ", cache_dir=str(tmp_path))
    assert info["symbol"] == "BTCUSDC"
    fg_res = info["pillar3_sentiment"]["fear_greed"]
    assert isinstance(fg_res, dict)
    assert fg_res["classification"] == "Extreme Fear"
    assert fg_res["age_formatted"] != "never"

    # 2. bapi_ws_health: string booleans ("false", "0") do not evaluate to True
    orig_avail = mark_ws_available(False)
    try:
        assert mark_ws_available("false") is False  # type: ignore[arg-type]
        assert mark_ws_available("true") is True  # type: ignore[arg-type]
        assert should_poll_for_manager(
            "CacheOrderManager",
            ws_only_mode="false",  # type: ignore[arg-type]
            ws_available="true",  # type: ignore[arg-type]
            ws_is_healthy="true",  # type: ignore[arg-type]
        ) is True
    finally:
        mark_ws_available(orig_avail)

    # 3. BinancePriceStream: whitespace/lowercase symbol normalization and invalid symbol rejection
    stream = BinancePriceStream(symbols=[" btcusdc ", None, ""])
    assert stream.running_symbols == {"BTCUSDC"}
    assert stream.add_symbol("") is False
    assert stream.add_symbol(" ethusdc ") is True
    assert "ETHUSDC" in stream.running_symbols
    stream._process_message('{"data": {"s": " btcusdc ", "c": "65000.5", "E": 1700000000000}}')
    assert stream.get_price(" btcusdc ") == 65000.5
    assert stream.get_price("") is None
    assert stream.remove_symbol(" btcusdc ") is True
    assert "BTCUSDC" not in stream.running_symbols

    # 4. StockYahooPricePlatform: whitespace ticker normalization
    yahoo = StockYahooPricePlatform(tickers=["NVDA", "TSLA"])
    assert yahoo.supports_symbol(" nvda ") is True
    assert yahoo.supports_symbol("   ") is False


def test_monitortrades_agy_paths_and_external_collectors_robustness(tmp_path, monkeypatch):
    """Verify monitortrades, orchestratorOS.agy.paths, and external collectors handle malformed payloads and non-numeric state values."""
    from types import SimpleNamespace
    import monitortrades as mt
    from orchestratorOS.agy import paths as agy_paths
    from intelligence.external.collectors.whale_positioning import WhalePositioningSnapshot
    from strategies.position_exit_manager import ExitAction, ExitDecision, PositionSnapshot

    # 1. WhalePositioningSnapshot.from_dict handles non-dict payload gracefully
    snap = WhalePositioningSnapshot.from_dict(None)  # type: ignore[arg-type]
    assert snap.symbol == ""
    assert snap.divergence_regime == "neutral"

    # 2. orchestratorOS.agy.paths handles bools, Path objects, and ~ in visible_conversations_dir
    assert agy_paths.expand_config_path(True) == ""  # type: ignore[arg-type]
    assert agy_paths.expand_config_path(tmp_path) == str(tmp_path)  # type: ignore[arg-type]
    vis_dir = agy_paths.visible_conversations_dir(str(tmp_path / ".gemini" / "antigravity-cli"))
    assert vis_dir.endswith(os.path.join(".gemini", "antigravity", "conversations"))

    # 3. monitortrades._extract_trade_uid and check_and_notify_executed_trades with malformed/tradeId items
    assert mt._extract_trade_uid(None) == ""
    assert mt._extract_trade_uid({"tradeId": 999}) == "999"
    assert mt._extract_trade_uid({"id": True}) == ""

    mt.reset_trades_monitor_state_for_test()
    monkeypatch.setattr(
        mt.apitrades,
        "get_trade_orders",
        lambda side, sym, maxage: ["not_a_dict", {"tradeId": "t100", "orderId": "o1", "price": "50000", "qty": "0.01", "side": "BUY"}],
    )
    mt.check_and_notify_executed_trades(symbols_list=["BTCUSDC"])
    assert "t100" in mt._notified_trade_ids

    # 4. monitortrades.get_available_qty rejects bools, negative numbers, and non-finite strings
    assert mt.get_available_qty("BTCUSDC", api=SimpleNamespace(free_balance=lambda base: True)) is None
    assert mt.get_available_qty("BTCUSDC", api=SimpleNamespace(free_balance=lambda base: -1.5)) is None
    assert mt.get_available_qty("BTCUSDC", api=SimpleNamespace(free_balance=lambda base: "bad")) is None
    assert mt.get_available_qty("BTCUSDC", api=SimpleNamespace(free_balance=lambda base: "1.25")) == 1.25

    # 5. monitortrades._place_guarded and _log_tp_hold_state / _log_sl_hold_state tolerate malformed numeric strings
    placed_calls = []
    dummy_inst = SimpleNamespace(symbol="BTCUSDC", place=lambda *a, **kw: placed_calls.append((a, kw)) or {"orderId": "1"})
    assert mt._place_guarded(dummy_inst, "BUY", 100.0, 0.5, "not_a_float") is True
    assert len(placed_calls) == 1

    pos_snap = PositionSnapshot(
        symbol="BTCUSDC",
        current_price=105.0,
        avg_buy_price=100.0,
        sellable_qty=1.0,
        current_time_s=1000.0,
    )
    dec = ExitDecision(action=ExitAction.HOLD, should_exit=False, reason="hold", metadata={"disaster_floor": "corrupt_floor"})
    monkeypatch.setitem(mt._TP_RIDE_STATE, "BTCUSDC", {"peak_price": "corrupt", "last_peak_time": "corrupt", "peak_gain": "corrupt"})
    monkeypatch.setitem(mt._SL_BOUNCE_STATE, "BTCUSDC", {"peak_bounce_price": "corrupt", "started_at": "corrupt"})
    mt._log_tp_hold_state("BTCUSDC", pos_snap, dec, 0.05)
    mt._log_sl_hold_state("BTCUSDC", pos_snap, dec, 0.05)
    assert mt._summarize_orders(["not_a_dict"]) == (0.0, 0.0, (1, 0.0, 0.0, 0, 0))


def test_monitororder_tradeall_and_assetguardian_edge_cases(monkeypatch):
    """Verify monitororder, tradeall, and assetguardian handle bool TTLs/quantities, string initial_prices, and whitespace symbols."""
    import monitororder as mo
    import tradeall as ta
    import assetguardian as ag

    # 1. monitororder.resolve_effective_order_ttl ignores boolean custom_ttl
    ttl = mo.resolve_effective_order_ttl("o1", "cid1", owner="tradeall", custom_ttl_map={"o1": True})
    assert ttl == min(mo.MODULE_ORDER_TTLS["tradeall"], mo.MAX_GLOBAL_ORDER_TTL_SEC)

    # 2. monitororder.monitor_open_orders_by_type handles non-dict order items, string initial_prices, and boolean origQty
    mo.reset_monitororder_state_for_test()
    mo.initial_sell_prices["o_str_px"] = "100.0"
    mo.initial_sell_prices["o_corrupt_px"] = "not_a_float"
    monkeypatch.setattr(
        mo.api,
        "get_open_orders",
        lambda side, sym: {
            "bad_item": "not_a_dict",
            "o_str_px": {"price": "100.0", "origQty": True, "clientOrderId": "TA_1"},
            "o_corrupt_px": {"price": "100.0", "origQty": "0.0", "clientOrderId": "TA_2"},
        },
    )
    monkeypatch.setattr(mo.api, "get_current_price", lambda sym: 100.0)
    mo.monitor_open_orders_by_type("BTCUSDC", "SELL")
    assert mo.initial_sell_prices["o_str_px"] == 100.0
    assert mo.initial_sell_prices["o_corrupt_px"] == 100.0

    # 3. tradeall._kalman_gate_blocks normalizes whitespace/lowercase symbols for configured coins
    if ta.KALMAN_MODES:
        sample_sym = next(iter(ta.KALMAN_MODES))
        monkeypatch.setattr(ta, "_kalman_ref", None)
        assert ta._kalman_gate_blocks(f" {sample_sym.lower()} ", "BUY")[0] is False

    # 4. assetguardian._complete_campaign_tier ignores invalid threshold and evaluate_symbol normalizes symbol
    saved_calls = []
    monkeypatch.setattr(ag, "_save_symbol_campaign", lambda s, k, st: saved_calls.append((s, k, st)))
    ag._complete_campaign_tier("BTCUSDC", "buy", {}, True)
    ag._complete_campaign_tier("BTCUSDC", "buy", {}, -5.0)
    assert saved_calls == []
    assert ag.evaluate_symbol(" unconfigured_sym ") is False


def test_position_exit_manager_rtrade_pair_and_bapi_placeorder_hardening():
    """Verify PositionExitManager, rtrade_pair, trailing_stop, and bapi_placeorder handle malformed inputs safely."""
    import pytest
    from binance_api import bapi_placeorder as bpo
    from binance_api.trailing_stop import TRAIL_PCT, TrailingStop
    from providers.strategy_executor import SubmissionRefused
    from strategies.position_exit_manager import (
        ExitAction,
        PositionExitManager,
        PositionSnapshot,
        ReentryAction,
        ReentrySnapshot,
    )
    from strategies.rtrade_pair import OrderTicket, PairCoordinator, PairPolicy

    # 1. PositionSnapshot / ReentrySnapshot geometry properties handle None/NaN/bool/corrupt strings safely
    bad_pos = PositionSnapshot(symbol="BTCUSDC", current_price=None, avg_buy_price=float("nan"))  # type: ignore[arg-type]
    assert bad_pos.is_valid_price is False
    assert bad_pos.price_change_ratio == 0.0
    assert bad_pos.gain_ratio == 0.0
    assert bad_pos.loss_ratio == 0.0
    assert bad_pos.bounce_ratio(float("nan")) == 0.0
    assert bad_pos.pullback_ratio(None) == 0.0  # type: ignore[arg-type]
    assert bad_pos.has_bounced(100.0, float("nan")) is False

    bad_re = ReentrySnapshot(symbol="BTCUSDC", current_price=100.0, last_sell_price="not_a_float")  # type: ignore[arg-type]
    assert bad_re.sale_drop_ratio == 0.0
    mgr = PositionExitManager()
    re_dec = mgr.evaluate_reentry(bad_re, custom_drop_pct=True)  # type: ignore[arg-type]
    assert re_dec.action == ReentryAction.ENTER
    assert re_dec.should_enter is True

    valid_pos = PositionSnapshot(symbol="BTCUSDC", current_price=94.0, avg_buy_price=100.0, current_time_s=1000.0)
    sl_dec = mgr.evaluate(valid_pos, sl_threshold=True, tp_threshold="invalid")  # type: ignore[arg-type]
    assert sl_dec.action == ExitAction.SELL

    # 2. rtrade_pair OrderTicket and PairCoordinator.from_state reject non-integer/overflow revisions and normalize side
    ticket = OrderTicket(order_id=" 123 ", side=" buy ", price="100.5", qty="0.1", revision=2.0)  # type: ignore[arg-type]
    assert ticket.order_id == "123"
    assert ticket.side == "BUY"
    assert ticket.revision == 2
    with pytest.raises(ValueError):
        OrderTicket(order_id="123", side="BUY", price=100.0, qty=0.1, revision=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OrderTicket(order_id="123", side="BUY", price=100.0, qty=0.1, revision=float("inf"))  # type: ignore[arg-type]

    policy = PairPolicy(adjustment_fraction=0.01)
    with pytest.raises(ValueError):
        PairCoordinator.from_state(
            None,
            policy,
            {
                "pair_id": "p1",
                "qty": 1.0,
                "start_side": "BUY",
                "phase": "quoting",
                "tickets": [],
                "limit_revisions": {"BUY": 1.5, "SELL": 0},
            },
        )

    # 3. bapi_placeorder.issue_account_cache_submit_permit rejects None/bool symbol or side before cache I/O
    with pytest.raises(SubmissionRefused):
        bpo.issue_account_cache_submit_permit(None, "BUY", 1.0, market=True)
    with pytest.raises(SubmissionRefused):
        bpo.issue_account_cache_submit_permit(True, "BUY", 1.0, market=True)
    with pytest.raises(SubmissionRefused):
        bpo.issue_account_cache_submit_permit("BTCUSDC", None, 1.0, market=True)

    # 4. TrailingStop.trail_pct_for normalizes symbol case/whitespace
    if TRAIL_PCT:
        sample_sym = next(iter(TRAIL_PCT))
        ts_dummy = object.__new__(TrailingStop)
        assert ts_dummy.trail_pct_for(f" {sample_sym.lower()} ") == TRAIL_PCT[sample_sym]


def test_cache_manager_item_validation_and_kraken_trailing_hardening():
    """Verify CacheTradeManager, CacheOrderManager, CacheCurrentPriceManager, and KrakenTrailing handle malformed inputs safely."""
    import pytest
    from cacheManager import CacheCurrentPriceManager, CacheOrderManager, CacheTradeManager
    from kraken.trailing_stop import TRAIL_PCT as KRAKEN_TRAIL_PCT, KrakenTrailing

    trade_mgr = object.__new__(CacheTradeManager)
    valid_trade = {
        "symbol": "BTCUSDC",
        "id": 101,
        "orderId": 202,
        "price": "50000.0",
        "qty": "0.01",
        "time": 1700000000000,
        "isBuyer": True,
    }
    assert trade_mgr._is_valid_item(valid_trade) is True
    assert trade_mgr._is_valid_item({**valid_trade, "id": 101.5}) is False
    assert trade_mgr._is_valid_item({**valid_trade, "orderId": 202.25}) is False
    assert trade_mgr._is_valid_item({**valid_trade, "time": 1700000000000.5}) is False

    order_mgr = object.__new__(CacheOrderManager)
    valid_order = {
        "orderId": 202,
        "price": 50000.0,
        "quantity": 0.01,
        "timestamp": 1700000000000,
        "side": "BUY",
        "_fillIds": ["101"],
    }
    assert order_mgr._is_valid_item(valid_order) is True
    assert order_mgr._is_valid_item({**valid_order, "orderId": 202.5}) is False
    assert order_mgr._is_valid_item({**valid_order, "timestamp": 1700000000000.5}) is False
    # Non-ASCII superscript digit "²" (\u00b2) must return False rather than raising ValueError
    assert order_mgr._is_valid_item({**valid_order, "_fillIds": ["\u00b2"]}) is False

    px_mgr = object.__new__(CacheCurrentPriceManager)
    with pytest.raises(ValueError, match="max_age_sec must be finite and positive"):
        px_mgr.get_cached_price_observation("BTCUSDC", None)  # type: ignore[arg-type]

    if KRAKEN_TRAIL_PCT:
        sample_asset = next(iter(KRAKEN_TRAIL_PCT))
        kt_dummy = object.__new__(KrakenTrailing)
        assert kt_dummy.trail_pct_for(f" {sample_asset.lower()} ") == KRAKEN_TRAIL_PCT[sample_asset]


def test_venue_market_data_and_client_hardening():
    """Verify Binance, Kraken, Hyperliquid, and T212 market_data/client helpers reject malformed inputs."""
    import importlib
    import importlib.util
    import pytest
    from binance_api import bapi
    from binance_api import market_data as b_md
    from kraken import market_data as k_md
    t212_md = importlib.import_module("212trading.market_data")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_hl_c = importlib.util.spec_from_file_location(
        "hl_client_local", os.path.join(repo_root, "hyperliquid", "hl_client.py")
    )
    hl_client = importlib.util.module_from_spec(spec_hl_c)
    spec_hl_c.loader.exec_module(hl_client)

    spec_hl_md = importlib.util.spec_from_file_location(
        "hl_market_data_local", os.path.join(repo_root, "hyperliquid", "market_data.py")
    )
    hl_md = importlib.util.module_from_spec(spec_hl_md)
    spec_hl_md.loader.exec_module(hl_md)

    # 1. Binance market_data: non-ASCII digits and invalid symbols
    with pytest.raises(ValueError, match="unsupported Binance interval"):
        b_md.parse_interval_minutes("\u00b2")
    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        b_md.fetch_ohlc_candles("   ", 60)

    # 2. Binance bapi helpers: whitespace/non-string asset or order_id
    assert bapi.get_free_balance("") is None
    assert bapi.get_free_balance(None) is None  # type: ignore[arg-type]
    assert bapi._convert_to_usdc(" usdc ", 25.5) == 25.5
    assert bapi._convert_to_usdc(None, 25.5) == 0.0
    assert bapi.cancel_order("BTCUSDC", "   ") is False
    assert bapi.check_order_filled("   ", "BTCUSDC") is False

    # 3. Kraken market_data: fractional float interval must raise ValueError
    with pytest.raises(ValueError, match="invalid interval"):
        k_md.fetch_ohlc_candles("XXBTZUSD", 1.5)  # type: ignore[arg-type]

    # 4. Hyperliquid market_data and hl_client: non-ASCII interval, fractional sz_decimals, invalid cancel coin
    with pytest.raises(ValueError, match="unsupported Hyperliquid interval"):
        hl_md.fetch_ohlc_candles("HYPE", "\u00b2")
    with pytest.raises(ValueError, match="sz_decimals must be a non-negative integer"):
        hl_client._round_px(25.1234, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sz_decimals must be a non-negative integer"):
        hl_client._round_px(25.1234, -1)

    dummy_hl = object.__new__(hl_client.HLClient)
    dummy_hl.exchange = object()
    assert dummy_hl.spot_order("@107", True, 1.0, 25.0, sz_decimals=1.5)[0] is False  # type: ignore[arg-type]
    assert dummy_hl.cancel("   ", 12345) is False
    assert dummy_hl.cancel(None, 12345) is False  # type: ignore[arg-type]

    # 5. 212trading market_data._chart short-circuits empty/non-string symbol without HTTP call
    assert t212_md._chart("") == (None, [])
    assert t212_md._chart("   ") == (None, [])
    assert t212_md._chart(None) == (None, [])  # type: ignore[arg-type]


def test_t212_strategy_order_manager_and_ipo_hardening():
    """Verify T212 strategy, order_manager, and ipo helpers handle malformed inputs safely."""
    import importlib
    t212_strat = importlib.import_module("212trading.strategy")
    t212_om = importlib.import_module("212trading.order_manager")
    t212_ipo = importlib.import_module("212trading.ipo")

    # 1. state_path_for handles bool/None/punctuation-only tickers
    assert t212_strat.state_path_for(True).endswith(".state_unknown.json")  # type: ignore[arg-type]
    assert t212_strat.state_path_for("---").endswith(".state_unknown.json")

    # 2. _find_open and _has_open are case-insensitive and tolerate missing side
    strat_dummy = object.__new__(t212_strat.Strategy)
    strat_dummy.s = {"orders": [{}, {"id": "1", "side": "sell", "limit": None, "ts": None}], "pending_submit": None}
    strat_dummy._clock = lambda: 1000.0
    strat_dummy.p = object.__new__(t212_strat.StratParams)
    strat_dummy.p.order_ttl_min = 5.0
    strat_dummy.p.entry_discount_pct = 1.0
    assert strat_dummy._has_open("SELL") is True
    assert strat_dummy._find_open("SELL") == {"id": "1", "side": "sell", "limit": None, "ts": None}
    assert strat_dummy._buy_is_stale({"id": "2", "side": "buy", "limit": None, "ts": None}, 100.0) is False

    # 3. order_manager rejects unsupported validity before touching client/disk
    assert t212_om._safe_marker_component(True) == "unknown"  # type: ignore[arg-type]
    assert t212_om.place_order_with_retry(None, "NVDA_US_EQ", 1.0, 100.0, "INVALID_VAL", dry_run=False) is False  # type: ignore[arg-type]
    assert t212_om.place_order_with_retry(None, "NVDA_US_EQ", 1.0, 100.0, "gtc", dry_run=True) is True  # type: ignore[arg-type]

    # 4. ipo.verify_instrument and _cmd_find_ticker reject empty/bool ticker or query
    assert t212_ipo.verify_instrument(None, "   ", "") is False  # type: ignore[arg-type]
    assert t212_ipo.verify_instrument(None, True, "") is False  # type: ignore[arg-type]
    assert t212_ipo._cmd_find_ticker(None, "   ") == 1  # type: ignore[arg-type]

    # 5. kraken_bot._cmd_find_pair and hyperliquid/strategy.py state_path_for & reconcile hardening
    import importlib.util
    from kraken import kraken_bot
    assert kraken_bot._cmd_find_pair(None, "   ") == 1  # type: ignore[arg-type]
    assert kraken_bot._cmd_find_pair(None, True) == 1  # type: ignore[arg-type]

    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_hl_s = importlib.util.spec_from_file_location(
        "hl_strategy_local", os.path.join(repo_root, "hyperliquid", "strategy.py")
    )
    hl_strat = importlib.util.module_from_spec(spec_hl_s)
    sys.modules[spec_hl_s.name] = hl_strat
    spec_hl_s.loader.exec_module(hl_strat)
    assert hl_strat.state_path_for(True, "---").endswith(".state_unknown_long.json")  # type: ignore[arg-type]
    hl_s_inst = hl_strat.Strategy(None, "HYPE", hl_strat.StratParams(), dry_run=True, replay_mode=True)
    assert hl_s_inst.reconcile(float("nan")) is False
    assert hl_s_inst.reconcile(-5.0) is False
    assert hl_s_inst.reconcile(True) is False  # type: ignore[arg-type]


def test_venue_replay_and_tooling_hardening(monkeypatch, tmp_path):
    import importlib.util
    import json
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. hyperliquid/price_analysis.py rejects fractional float block/point counts
    spec_hl_pa = importlib.util.spec_from_file_location(
        "hl_pa_local", os.path.join(repo_root, "hyperliquid", "price_analysis.py")
    )
    hl_pa = importlib.util.module_from_spec(spec_hl_pa)
    sys.modules[spec_hl_pa.name] = hl_pa
    spec_hl_pa.loader.exec_module(hl_pa)
    ts = [i * 3600 for i in range(100)]
    px = [100.0 + i * 0.5 for i in range(100)]
    assert hl_pa.detect_long_term_trend(ts, px, min_consecutive_blocks=1.5) is None
    assert hl_pa.detect_long_term_trend(ts, px, noise_tolerance=0.5) is None
    assert hl_pa.detect_long_term_trend(ts, px, min_points_per_window=2.5) is None

    # 2. hyperliquid/replay.py sanitizes currency on empty OHLC
    spec_hl_rp = importlib.util.spec_from_file_location(
        "hl_rp_local", os.path.join(repo_root, "hyperliquid", "replay.py")
    )
    hl_rp = importlib.util.module_from_spec(spec_hl_rp)
    sys.modules[spec_hl_rp.name] = hl_rp
    spec_hl_rp.loader.exec_module(hl_rp)
    assert hl_rp.run_replay([], [], [], [], {"currency": " usdc "})["account_currency"] == "USDC"
    assert hl_rp.run_replay([], [], [], [], {"currency": True})["account_currency"] == "USDC"
    assert hl_rp.run_replay([], [], [], [], {"currency": None})["account_currency"] == "USDC"

    # 3. strategies/spot_replay.py handles numeric-string OHLC and malformed warmup_ohlc
    from strategies import spot_engine, spot_replay
    p_spot = spot_engine.StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=0.2,
        dca_amount=50.0,
        dca_drop_pct=2.0,
        check_minutes=1,
        takeprofit_pct=2.0,
        max_budget=1000.0,
        max_dca_buys=5,
        enable_takeprofit=True,
        order_ttl_min=10,
    )
    res_spot = spot_replay.run_replay(
        [("100.0", "105.0", "95.0", "102.0"), ("102.0", "106.0", "101.0", "104.0")],
        p_spot,
        bar_minutes=240.0,
        warmup_ohlc=[("bad",), (100, 101, 99, True), (100, 101, 99, -5.0), (100, 101, 99, 98.5)],
    )
    assert res_spot["initial_capital"] > 0

    # 4. kraken/shadow_live.py sanitizes pair path, returns int anchor, and skips h < l history bars
    from kraken import shadow_live as k_sl
    monkeypatch.setattr(k_sl, "LOG_DIR", str(tmp_path))
    assert k_sl._safe_pair("HYPE/USD") == "HYPE_USD"
    anchor_val = k_sl._get_anchor("HYPE/USD", 240, 1700000000.0)  # type: ignore[arg-type]
    assert isinstance(anchor_val, int) and anchor_val == 1700000000
    hist_path = k_sl._history_path("HYPE/USD", 240)
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump([[1700000000, 10.0, 9.0, 9.5, 9.2], [1700014400, 10.0, 12.0, 9.5, 11.0]], fh)
    assert k_sl._load_history("HYPE/USD", 240) == [(1700014400, 10.0, 12.0, 9.5, 11.0)]

    # 5. kraken/backtest.py handles None interval/fee in _bt_params and simulate
    from kraken import backtest as k_bt
    bt_p = k_bt._bt_params({"interval": None})
    assert bt_p.entry_amount > 0

    # 6. 212trading/price_alert.py push rejects blank/whitespace topic
    spec_pa = importlib.util.spec_from_file_location(
        "t212_pa_local", os.path.join(repo_root, "212trading", "price_alert.py")
    )
    t212_pa = importlib.util.module_from_spec(spec_pa)
    sys.modules[spec_pa.name] = t212_pa
    spec_pa.loader.exec_module(t212_pa)
    assert t212_pa.push("   ", "NVDA at 100", "body") is False

    # 7. cachepaths, tradeall_price_archiver._symbols, kraken backtest_adaptive/backtest_trail, and binance_api.trailing_stop
    import cachepaths
    import pytest
    import tradeall_price_archiver
    from kraken import backtest_adaptive as k_ba, backtest_trail as k_bt_tr
    from binance_api import trailing_stop as b_ts

    monkeypatch.setenv("MPTRADE_CACHEDB_DIR", "   ")
    monkeypatch.setenv("BINANCE_CACHE_DIR", str(tmp_path))
    assert cachepaths.get_cachedb_dir() == str(tmp_path)
    assert cachepaths.cache_path("./cachedb/foo.json") == os.path.join(str(tmp_path), "foo.json")
    assert cachepaths.legacy_filenames_for(None) == ()  # type: ignore[arg-type]
    assert cachepaths.legacy_filenames_for(True) == ()  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        tradeall_price_archiver._symbols(None)
    with pytest.raises(ValueError):
        tradeall_price_archiver._symbols(True)

    bars_sample = [(100.0, 102.0, 99.0, 101.0), (101.0, 103.0, 100.0, 102.0)]
    assert k_ba.simulate_variant(bars_sample, {"drop_fallback": None, "reentry_drop": None, "fee": None, "interval": None}, [1.5, 1.5])["initial_capital"] > 0
    assert k_bt_tr.simulate_trail(bars_sample, {"trail": None, "reentry_drop": None, "fee": None, "interval": None})["initial_capital"] > 0
    assert k_bt_tr.bh([(100.0, 101.0, 99.0, True)]) == 0.0  # type: ignore[list-item]
    assert k_bt_tr.bh([(100.0,)]) == 0.0  # type: ignore[list-item]

    # 8. providers.base.extract_order_id and binance_api.bapi_allorders/bapi_trades hardening
    from types import SimpleNamespace
    from providers.base import extract_order_id
    from binance_api import bapi_allorders, bapi_trades

    assert extract_order_id({"orderId": "", "id": "999"}) == "999"
    assert extract_order_id({"orderId": "   ", "txid": ["", "TX42"]}) == "TX42"
    assert extract_order_id({"orderId": float("nan"), "id": 777}) == "777"
    assert extract_order_id({"orderId": float("nan")}) is None

    agg = bapi_trades.aggregate_trades([
        {"orderId": "", "id": 501, "symbol": "BTCUSDC", "price": "100.0", "qty": "1.0", "time": 1000, "isBuyer": True},
        {"orderId": float("nan"), "id": 502, "symbol": "BTCUSDC", "price": "200.0", "qty": "1.0", "time": 2000, "isBuyer": False},
    ])
    assert [row["orderId"] for row in agg] == ["501", "502"]

    bad_client = SimpleNamespace(get_my_trades=lambda **kw: [{"id": 12.5}, {"id": 13.5}])
    with pytest.raises(RuntimeError):
        bapi_allorders.paginate_my_trades(bad_client, "BTCUSDC", 1000, limit=2, strict=True)

    # 9. bapi_ws_normalize, order_id_context, and bapi_ws stop self-join hardening
    import threading
    from binance_api import bapi_ws, bapi_ws_normalize, order_id_context

    assert bapi_ws_normalize.parse_order_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "X": "FILLED", "t": 12.5, "i": 1, "L": "50000", "l": "0.1"}
    ) is None
    assert bapi_ws_normalize.parse_trade_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "S": "BUY", "t": 12.5, "i": 1, "L": "50000", "l": "0.1"}
    ) is None
    assert bapi_ws_normalize.parse_trade_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "S": "BUY", "t": "-5.5", "i": 1, "L": "50000", "l": "0.1"}
    ) is None
    assert bapi_ws_normalize.parse_trade_execution_report(
        {"s": "BTCUSDC", "x": "TRADE", "S": "BUY", "t": None, "i": "   ", "L": "50000", "l": "0.1"}
    ) is None

    assert order_id_context.resolve_order_owner(float("-inf")) == "unspecified"  # type: ignore[arg-type]
    assert order_id_context._derive_process_name(True) == "PYTHON"

    ws_base = bapi_ws.BinanceWSBase()
    ws_base._thread = threading.current_thread()
    assert ws_base.stop(timeout=0.01) is True



def test_root_modules_batch4_hardening():
    import assetguardian
    import liquidation_signal
    import log
    import order_guard
    import priceAnalysis
    import pricewindow
    import utils

    # order_guard.select_extreme_reference and _aggregate_reference_prices
    assert order_guard.select_extreme_reference([None, "bad", True, float("nan")], "BUY") is None
    assert order_guard.select_extreme_reference(None, "SELL") is None
    assert order_guard.select_extreme_reference([100.0, "bad", 95.0], "BUY", strict=True) == 95.0
    assert order_guard._aggregate_reference_prices(
        [("bad_qty", 100.0), (2.0, "110.0"), ("not_a_tuple",)],
        [("bad_ts", 1.0, 100.0), (1000.0, 2.0, 110.0)],
        "SELL",
        "average",
    ) == 110.0

    # utils.slope and decrese_value_by_increment_exp
    assert utils.slope(True, 0, 10, 5) == 0.0
    assert utils.slope(0, 5, 10, 5) == 0.0
    assert utils.slope(0, 0, 10, 5) == 2.0
    v1, v2 = utils.decrese_value_by_increment_exp(-1, 100.0)
    assert v1 > 0.0 and v2 > 0.0
    assert utils.decrese_value_by_increment_exp(True, 100.0) == (0.0, 0.0)
    v1_big, v2_big = utils.decrese_value_by_increment_exp(-1e6, 100.0)
    assert v1_big >= 0.0 and v2_big >= 0.0

    # priceAnalysis.slope_tolerance_per_ and get_trade_weight
    assert priceAnalysis.slope_tolerance_per_("BTCUSDC", None) == 0.0005
    assert priceAnalysis.slope_tolerance_per_("BTCUSDC", True) == 0.0005
    assert priceAnalysis.slope_tolerance_per_("BTCUSDC", 100000.0, base_tolerance=True) == 150.0
    t_seq, w_seq = priceAnalysis.get_trade_weight(
        100, 50, "up", "BUY", exceed_percent=True, max_against_trend=None, peak_weight=float("nan"), min_weight=False
    )
    assert len(w_seq) > 0 and float(w_seq[0]) > 0.0

    # pricewindow.PriceTrendAnalyzer and PriceWindow._sample_rate_from_entries
    pta = pricewindow.PriceTrendAnalyzer([None, True, 100.0, "bad", 102.0, 104.0])
    assert pta.prices == [100.0, 102.0, 104.0]
    tl, slp, rv = pta.linear_regression_trend()
    assert slp == 2.0
    assert len(pta.exponential_moving_average(span=3)) == 3
    assert pricewindow.PriceWindow._sample_rate_from_entries([None, "bad", [True, 100.0], [1000, 100.0], [3000, 101.0]]) == 2.0

    # assetguardian threshold helpers
    assert assetguardian._effective_portfolio_growth_threshold(True) == assetguardian.PORTFOLIO_GROWTH_PERCENT
    assert assetguardian._trigger_threshold_for_sell_tier(True, sell_mult=None, portfolio_growth_pct=float("nan")) >= assetguardian.SELL_REARM_GROWTH_PERCENT + 2.0
    assert assetguardian._trigger_threshold_for_buy_tier(None, buy_mult=True) >= assetguardian.RECOVERY_RESET_PERCENT + 0.5

    # liquidation_signal.LiquidationSignal
    ls = liquidation_signal.LiquidationSignal(symbols="btcusdt, ethusdt", retention_sec=True)
    assert ls._symbols == ["BTCUSDT", "ETHUSDT"]
    ls._ingest('{"topic":"allLiquidation.BTCUSDT","data":123}')
    ls._ingest('{"topic":"allLiquidation.BTCUSDT","data":[{"s":"BTCUSDT","S":"Buy","v":"1.5","p":"60000"}]}')
    sig = ls.get_signal("BTCUSDT", window_sec=True)
    assert sig["long_liq_usd"] == 90000.0

    # log._patched_stderr_write returns int count
    written = log._patched_stderr_write("")
    assert isinstance(written, int)


def test_batch5_hardening():
    import importlib.util
    import numpy as np
    from binance_api import binanceWSeventsFake
    from intelligence.internal.state import survival
    from notify_engine import server as notify_server
    import priceAnalysis

    from pathlib import Path
    hl_pa_path = Path(__file__).resolve().parent.parent / "hyperliquid" / "price_analysis.py"
    spec = importlib.util.spec_from_file_location("hl_price_analysis_local", hl_pa_path)
    assert spec is not None and spec.loader is not None
    hl_pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl_pa)

    # 1. survival.block_slopes, episodes, survival_report with bad inputs and duplicate timestamps
    bt, bs = survival.block_slopes("bad", "bad", 24, 8)
    assert len(bt) == 0 and len(bs) == 0
    # Collinear/constant timestamps in a window must not raise LinAlgError
    ts_collinear = [1000.0] * 10 + [1000.0 + i * 3600.0 for i in range(1, 30)]
    px_collinear = [100.0] * 10 + [100.0 + i * 2.0 for i in range(1, 30)]
    bt2, bs2 = survival.block_slopes(ts_collinear, px_collinear, 6, 2)
    assert len(bt2) == len(bs2) and len(bt2) > 0
    assert survival.episodes(bt2, bs2, 6, noise_tolerance=float("nan"), min_confirm="bad") is not None
    assert survival.survival_report([None, True, "bad", 10.0], "BTC", [1, 2], horizon_h=True) is None

    # 2. notify_engine.server coercion of non-str title/source/bot_name
    assert notify_server._category_for_title_and_source(12345, True) == "TRADES"  # type: ignore[arg-type]
    assert notify_server._resolve_provider_label(123, None) == "123"  # type: ignore[arg-type]
    assert notify_server._resolve_provider_label(True, "kraken_bot") == "Kraken"  # type: ignore[arg-type]

    # 3. binanceWSeventsFake string delta, non-dict execution report, and trades_cache cap
    binanceWSeventsFake.balances_cache.clear()
    binanceWSeventsFake.orders_cache.clear()
    binanceWSeventsFake.trades_cache.clear()
    binanceWSeventsFake.handle_balance_update("USDC", "12.5", 1700000000000)
    assert binanceWSeventsFake.balances_cache["USDC"] == 12.5
    binanceWSeventsFake.handle_execution_report("not_a_dict")
    binanceWSeventsFake.handle_execution_report(
        {"s": "BTCUSDC", "i": 99, "X": "FILLED", "x": "TRADE", "S": "BUY", "p": "bad", "q": "1.0", "z": "1.0", "L": "65000.5", "l": "0.5"}
    )
    assert binanceWSeventsFake.orders_cache[99]["price"] == 0.0
    assert binanceWSeventsFake.trades_cache[-1]["price"] == 65000.5
    binanceWSeventsFake.handle_account_position(["not_a_dict", {"a": "BTC", "f": "1.25", "l": "bad"}])
    assert binanceWSeventsFake.balances_cache["BTC"] == 1.25

    # 4. detect_long_term_trend with non-iterable/str/bool/mixed elements in both priceAnalysis and hyperliquid.price_analysis
    assert hl_pa.detect_long_term_trend("12345", "12345") is None
    assert hl_pa.detect_long_term_trend(12345, 12345) is None
    assert priceAnalysis.detect_long_term_trend("12345", "12345") is None
    assert priceAnalysis.detect_long_term_trend(np.array([True, False, True]), np.array([1.0, 2.0, 3.0])) is None
    assert priceAnalysis.detect_long_term_trend([1000.0, None, "bad", True, 2000.0], [100.0, 101.0, 102.0, 103.0, 104.0]) is None


def test_cooldown_and_market_monitor_hardening(tmp_path):
    import threading
    from types import SimpleNamespace
    from lock.cooldown import Cooldown
    from lock import trade_cooldown
    from market_monitor.new_coins_discovery import NewCoinsMonitor
    from market_monitor.pricechecker import PriceChecker
    from market_monitor.pricefetcher import CacheAllPriceFetcherManager

    # 1. Cooldown group methods strip group_id and member whitespace
    cd = Cooldown("test_cd", base_dir=str(tmp_path))
    allowed, _ = cd.reserve_group_member("BTCUSDC", 60, "  grp1  ", "  buy  ")
    assert allowed is True
    cd.commit_group_member("BTCUSDC", "grp1", "BUY", order_id=101)
    state = cd.get("BTCUSDC")
    assert state["group_id"] == "grp1"
    assert state["group_committed"] == ["BUY"]
    assert cd.release_group_member("BTCUSDC", "  grp1 ", " buy ", keep_group=False) is True

    # 2. trade_cooldown.release_pair_leg rejects invalid pair_id and strips whitespace
    assert trade_cooldown.release_pair_leg("BTCUSDC", None, "BUY") is False
    assert trade_cooldown.release_pair_leg("BTCUSDC", True, "BUY") is False
    assert trade_cooldown.release_pair_leg("BTCUSDC", "   ", "BUY") is False

    # 3. PriceChecker._is_dynamic_symbol rejects NaN/bool/non-positive timestamps and stop_monitoring avoids self-join
    dummy_cache = SimpleNamespace(
        original_symbols=["BTCUSDC"],
        symbol_added_time={"NAN_SYM": float("nan"), "BOOL_SYM": True, "NEG_SYM": -5.0, "OK_SYM": 1700000000.0},
    )
    checker = PriceChecker(dummy_cache)
    assert checker._is_dynamic_symbol("NAN_SYM") is False
    assert checker._is_dynamic_symbol("BOOL_SYM") is False
    assert checker._is_dynamic_symbol("NEG_SYM") is False
    assert checker._is_dynamic_symbol("OK_SYM") is True
    checker._thread = threading.current_thread()
    checker._running = True
    checker.stop_monitoring()
    assert checker._running is False

    # 4. NewCoinsMonitor.stop_monitoring avoids self-join
    ncm = NewCoinsMonitor.__new__(NewCoinsMonitor)
    ncm._running = True
    ncm._thread = threading.current_thread()
    ncm.stop_monitoring()
    assert ncm._running is False

    # 5. CacheAllPriceFetcherManager.cleanup_old_symbols evicts NaN/non-positive added_time
    pf = CacheAllPriceFetcherManager.__new__(CacheAllPriceFetcherManager)
    pf.lock = threading.RLock()
    pf.symbols = ["NANSYM"]
    pf.original_symbols = ["NANSYM"]
    pf.active_symbols = {"NANSYM"}
    pf.symbol_added_time = {"NANSYM": float("nan"), "NEGSYM": -1.0}
    pf.symbol_preferred_source = {}
    removed = pf.cleanup_old_symbols(max_age_days=7)
    assert set(removed) == {"NANSYM", "NEGSYM"}
    assert pf.symbol_added_time == {}


def test_batch6_hardening(tmp_path):
    import json
    from types import SimpleNamespace
    from market_monitor import alerts_config
    from market_monitor.new_coins_discovery import NewCoinsMonitor
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    from visuals import drawprices
    import cacheManager as cm

    # 1. AutonomousIntentGateway._get_live_price reads latest entry ([-1]) from cache_current_price.json
    cache_dir = tmp_path / "cachedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_current_price.json").write_text(
        json.dumps({"items": {"BTCUSDC": [[1000, 50000.0], [2000, 62500.0], [3000, None]]}}),
        encoding="utf-8",
    )
    gw = AutonomousIntentGateway.__new__(AutonomousIntentGateway)
    gw.cache_dir = str(cache_dir)
    dummy_inst = SimpleNamespace(symbol="BTCUSDC", base="BTC", _provider=SimpleNamespace())
    assert gw._get_live_price(dummy_inst) == 62500.0

    # 2. alerts_config.resolve handles missing keys, non-dict per_coin, and string is_dynamic
    assert alerts_config.resolve(None, "BTCUSDC", "false") == {"up_percent": 4.1, "down_percent": 7.5}  # type: ignore[arg-type]
    assert alerts_config.resolve(
        {"default": {"up_percent": 5.0, "down_percent": 8.0}, "dynamic": {"up_percent": 15.0, "down_percent": 25.0}},
        "BTCUSDC",
        "false",  # type: ignore[arg-type]
    ) == {"up_percent": 5.0, "down_percent": 8.0}

    # 3. NewCoinsMonitor partial auto_add_thresholds and symbol_added_time=None
    added_calls = []
    pm = SimpleNamespace(
        original_symbols=[],
        active_symbols=set(),
        max_monitored_symbols=10,
        symbol_added_time=None,
        add_symbol=lambda sym, preferred_source=None: (added_calls.append(sym), True)[1],
    )
    ncm = NewCoinsMonitor.__new__(NewCoinsMonitor)
    ncm.price_monitor = pm
    assert ncm.should_keep_monitoring("NEWCOIN") is True
    assert ncm.add_new_coin_to_watchlist(
        {"symbol": "NEWCOIN", "name": "New Coin", "price": 1.25, "volume_24h": 50000},
        auto_add_thresholds={"min_volume_24h": 20000},
    ) is True
    assert added_calls == ["NEWCOIN"]

    # 4. visuals.drawprices.get_cache_for_symbol with dict-backed entry.cache and forecast_linear
    old_mgr = getattr(cm, "price_cache_manager", None)
    try:
        cm.price_cache_manager = {"BTCUSDC": SimpleNamespace(cache={"BTCUSDC": [[1700000000000, 60000.0], [1700000060000, "bad"]]})}
        assert drawprices.get_cache_for_symbol("BTCUSDC") == [(1700000000000, 60000.0)]
    finally:
        cm.price_cache_manager = old_mgr
    assert len(drawprices.forecast_linear(None, [100.0, None, "bad", True, 110.0], horizon="5")) == 5  # type: ignore[arg-type]
    assert len(drawprices.forecast_linear(None, "bad", horizon=5)) == 0  # type: ignore[arg-type]


def test_batch7_hardening():
    import sys
    from offline.backtests.datasets import normalize_record
    from offline.backtests.metrics import calculate_performance_metrics
    from offline.backtests.walk_forward import summarize_test_windows
    from offline.research.chop_weight_backtest import run_single_simulation
    from offline.runners.walk_forward_compare import _load_hl_replay, _load_t212_replay

    # 1. datasets.normalize_record accepts integer-valued float string timestamp and rejects fractional/non-finite
    rec = normalize_record({"timestamp": "1700000000.0", "open": 100, "high": 105, "low": 99, "close": 102})
    assert rec["timestamp"] == 1700000000
    with pytest.raises(ValueError):
        normalize_record({"timestamp": "1700000000.5", "open": 100, "high": 105, "low": 99, "close": 102})
    with pytest.raises(ValueError):
        normalize_record({"timestamp": "nan", "open": 100, "high": 105, "low": 99, "close": 102})

    # 2. walk_forward.summarize_test_windows rejects bool/non-finite return_pct, max_drawdown_pct, buy_hold_return_pct
    with pytest.raises(ValueError):
        summarize_test_windows([
            {"return_pct": True, "max_drawdown_pct": 1.0, "buy_hold_return_pct": 2.0, "cycles": 1, "fills": 2}
        ])
    with pytest.raises(ValueError):
        summarize_test_windows([
            {"return_pct": float("nan"), "max_drawdown_pct": 1.0, "buy_hold_return_pct": 2.0, "cycles": 1, "fills": 2}
        ])
    with pytest.raises(ValueError):
        summarize_test_windows([
            {"return_pct": 1.0, "max_drawdown_pct": 1.0, "buy_hold_return_pct": float("inf"), "cycles": 1, "fills": 2}
        ])

    # 3. metrics.calculate_performance_metrics rejects bool/non-finite in equity_curve and trade_pnls
    with pytest.raises(ValueError):
        calculate_performance_metrics([1000.0, True], initial_capital=1000.0)
    with pytest.raises(ValueError):
        calculate_performance_metrics([1000.0, float("nan")], initial_capital=1000.0)
    with pytest.raises(ValueError):
        calculate_performance_metrics([1000.0, 1050.0], initial_capital=1000.0, trade_pnls=[50.0, False])

    # 4. chop_weight_backtest.run_single_simulation validates base_notional, tp_pct, stop_pct, T_days
    with pytest.raises(ValueError):
        run_single_simulation("BTCUSDC", [(1.0, 100.0)], None, "CHOP_0.03", base_notional=float("nan"))
    with pytest.raises(ValueError):
        run_single_simulation("BTCUSDC", [(1.0, 100.0)], None, "CHOP_0.03", tp_pct=-0.01)

    # 5. walk_forward_compare._load_hl_replay and _load_t212_replay preserve sys.path
    orig_path = list(sys.path)
    _load_hl_replay()
    assert list(sys.path) == orig_path
    _load_t212_replay()
    assert list(sys.path) == orig_path


def test_batch8_hardening():
    import numpy as np
    from offline.research.ml_forecast import forecast as ml_forecast
    from offline.research.ml_forecast.priceprediction import PricePrediction
    from offline.research.monitortrades_backtest.replay_trend_source import DynamicReplayTrendSource

    # 1. ml_forecast.walk_forward and live_forecast handle empty and single-class datasets
    empty_X = np.empty((0, len(ml_forecast.FEATURES)), dtype=float)
    empty_y_dir = np.empty(0, dtype=int)
    empty_y_mag = np.empty(0, dtype=float)
    wf_empty = ml_forecast.walk_forward(empty_X, empty_y_dir, empty_y_mag)
    assert wf_empty["n_test"] == 0
    lf_empty = ml_forecast.live_forecast(np.array([100.0, 101.0]), empty_X, empty_y_dir, empty_y_mag, wf_empty, "BTCUSDC")
    assert lf_empty["proba_up"] == 0.5

    single_X = np.ones((20, len(ml_forecast.FEATURES)), dtype=float)
    single_y_dir = np.ones(20, dtype=int)
    single_y_mag = np.full(20, 0.01, dtype=float)
    wf_single = ml_forecast.walk_forward(single_X, single_y_dir, single_y_mag, train_frac=0.5, refit_every=10)
    assert wf_single["logit"] == 1.0
    assert wf_single["boost"] == 1.0
    lf_single = ml_forecast.live_forecast(np.linspace(100.0, 120.0, 300), single_X, single_y_dir, single_y_mag, wf_single, "BTCUSDC")
    assert lf_single["proba_up"] == 1.0
    assert lf_single["trend"] == "up"

    # 2. PricePrediction sanitizes invalid/NaN/bool constructor args
    pp = PricePrediction(window_size=float("nan"), hidden_size=True, lr=-1.0, epochs=False)  # type: ignore[arg-type]
    assert pp.window_size == 10
    assert pp.epochs == 30

    # 3. DynamicReplayTrendSource handles bool/NaN min_agree and epsilon_k
    dr = DynamicReplayTrendSource(["BTCUSDC"])
    for idx, px in enumerate([100.0, 101.0, 102.0, 103.0]):
        dr.advance("BTCUSDC", 1000.0 + idx * 10.0, px)
    assert isinstance(dr.is_trend_up_confirmed("BTCUSDC", [30.0, 60.0], min_agree=float("nan")), bool)  # type: ignore[arg-type]
    assert isinstance(dr.should_wait("BTCUSDC", "BUY", 60.0, use_noise_gate=True, epsilon_k=float("nan")), bool)


def test_batch9_hardening(tmp_path):
    import pandas as pd
    from offline.research.tradeall_trigger_gate.experiment_rsi_bollinger import MeanReversionTracker
    from offline.research.tradeall_trigger_gate.experiment_rsi_dca_filter import RsiGate
    from offline.legacy_tools import convertor, gen_cryptography, test as legacy_test, test2 as legacy_test2

    # 1. MeanReversionTracker sanitizes bool/NaN/zero bb_k and avoids ZeroDivisionError
    mrt = MeanReversionTracker(
        "test", "BTCUSDC", mode="both",
        rsi_period=float("nan"), rsi_low=True, rsi_high=False,
        bb_period=float("inf"), bb_k=0.0, pb_low=float("nan"), pb_high=float("nan"),
    )
    assert mrt.rsi_period == 14
    assert mrt.bb_period == 20
    assert mrt.bb_k == 2.0
    for px in [100.0 + i * 0.5 for i in range(25)]:
        mrt.update_sign(px)
    assert mrt.sign in (-1, 0, 1)

    # 2. RsiGate sanitizes bool/NaN period and low
    rg = RsiGate(period=float("nan"), low=True)
    assert rg.period == 14
    assert rg.low == 35.0
    rg.update(True)
    assert rg.prev is None

    # 3. legacy_tools convertor, gen_cryptography, test, test2 are import-safe and handle NaN
    p_priv = tmp_path / "priv.pem"
    p_pub = tmp_path / "pub.pem"
    pub_b64 = gen_cryptography.generate_ed25519_keypair(str(p_priv), str(p_pub))
    assert isinstance(pub_b64, str) and len(pub_b64) > 10
    assert p_priv.exists() and p_pub.exists()

    df = pd.DataFrame({"close": [100.0] * 60 + [float("nan")] * 10 + [90.0] * 80})
    c_drops, t_int, prob = legacy_test.analyze_drops(df)
    assert t_int > 0 and prob >= 0.0
    res2 = legacy_test2.analyze_drops_with_distribution(df, show_plot=False)
    assert len(res2) == 6


def test_batch10_archive_hardening(tmp_path):
    from market_regime import MarketRegimeEvaluator, MarketRegimeService
    from archive.market_regime_backtest import evaluate_forward_classifications
    from archive.monitortrades_legacy import ProcentDistributor, adjust_monitor_interval
    from archive.tradeCacheManager import TradeCacheManager

    evaluator = MarketRegimeEvaluator()
    bull = evaluator.evaluate({"gradient_recent": 0.6, "epsilon": 0.1})
    decision = MarketRegimeService().compose(bull, bull)
    with pytest.raises(ValueError):
        evaluate_forward_classifications([(decision, True)])

    assert adjust_monitor_interval(20, 5, 0, -10) == 5
    with pytest.raises(ValueError):
        ProcentDistributor(0, 600, float("nan"), 0.01)
    with pytest.raises(ValueError):
        ProcentDistributor(0, 600, 0.1, True)
    pd_inst = ProcentDistributor(0, 600, 0.1, 0.01)
    pd_inst.update_tick(passed=1, half_life_duration=0)
    assert pd_inst.max_procent == 0.1

    tcm = TradeCacheManager(str(tmp_path / "nonexistent_cache.json"))
    assert tcm.last_fetch_time_per_symbol == {}


def test_batch11_verify_tools_and_watchdog_hardening(tmp_path, monkeypatch):
    import sys
    from verify_tools import check_cache_coherence, translation_audit
    from orchestratorOS.livecheck import os_resources_watchdog as orw

    # 1. translation_audit skips myenv
    assert "myenv" in translation_audit.SKIP_DIRS

    # 2. check_cache_coherence.expected_symbols does not duplicate sys.path
    orig_len = len(sys.path)
    check_cache_coherence.expected_symbols()
    check_cache_coherence.expected_symbols()
    assert len(sys.path) == orig_len

    # 3. os_resources_watchdog handles proc dict missing pid/command without KeyError
    monkeypatch.setattr(orw, "STATE_FILE", tmp_path / "rw_state.json")
    monkeypatch.setattr(orw.wc, "alert", lambda *a, **kw: True)
    procs = {"999:1": {"rss_mb": orw.MEMORY_THRESHOLD_MB + 100.0}}
    for step in range(orw.CONSECUTIVE + 1):
        orw.check_once(now=1000.0 + step * 60.0, total_ticks=1000 + step * 100, processes=procs, cpu_count=1)


def test_batch12_verify_tools_pnl_and_facade_hardening(monkeypatch, capsys):
    from types import SimpleNamespace
    from verify_tools import pnl_report, portfolio_snapshot, verify_account_facade

    # 1. verify_account_facade.position_from ignores buy orders with zero/negative price or zero qty
    avg, net = verify_account_facade.position_from(
        [{"price": 100.0, "qty": 2.0}, {"price": 0.0, "qty": 5.0}, {"price": 50.0, "qty": 0.0}],
        [{"qty": 0.5}, {"qty": 0.0}],
    )
    assert avg == 100.0
    assert net == 1.5

    # 2. portfolio_snapshot._is_paper handles boolean strings and ints consistently
    assert portfolio_snapshot._is_paper("false") is False
    assert portfolio_snapshot._is_paper("true") is True
    assert portfolio_snapshot._is_paper(False) is False
    assert portfolio_snapshot._is_paper(True) is True
    assert portfolio_snapshot._is_paper(0) is False
    assert portfolio_snapshot._is_paper(1) is True

    # 3. pnl_report.per_bot skips orders where both cummulativeQuoteQty and price are zero
    monkeypatch.setattr(
        pnl_report.bapi,
        "client",
        SimpleNamespace(
            get_all_orders=lambda **kw: [
                {"clientOrderId": "TA_1", "side": "BUY", "executedQty": "2.0", "cummulativeQuoteQty": "200.0", "price": "100.0"},
                {"clientOrderId": "TA_2", "side": "BUY", "executedQty": "8.0", "cummulativeQuoteQty": "0.0", "price": "0.0"},
                {"clientOrderId": "TA_3", "side": "SELL", "executedQty": "2.0", "cummulativeQuoteQty": "240.0", "price": "120.0"},
            ]
        ),
    )
    pnl_report.per_bot("BTCUSDC")
    out = capsys.readouterr().out
    assert "+40.00" in out


def test_batch13_agy_summary_store_topics_and_ui_mirror_safety(tmp_path):
    import json
    import pytest
    from orchestratorOS.agy import summary_store, topics, ui_mirror

    # 1. delete_threads refuses empty/dot/traversal cids and never wipes root brain/ dir
    cli_base = tmp_path / "antigravity-cli"
    brain_keep = cli_base / "brain" / "keep_cid"
    brain_keep.mkdir(parents=True)
    (brain_keep / "artifact.md").write_text("safe", encoding="utf-8")
    cnt, freed = summary_store.delete_threads({"", "  ", ".", "..", "../outside", "a/b"}, str(cli_base), dry_run=False)
    assert (cnt, freed) == (0, 0)
    assert (brain_keep / "artifact.md").exists()

    # dry_run="true" does not delete valid cid
    cnt2, freed2 = summary_store.delete_threads({"keep_cid"}, str(cli_base), dry_run="true")  # type: ignore[arg-type]
    assert cnt2 == 1 and freed2 > 0
    assert (brain_keep / "artifact.md").exists()

    # 2. resolve_project_ids ignores unhashable id/name values in project JSON files
    proj_dir = tmp_path / "config" / "projects"
    proj_dir.mkdir(parents=True)
    (proj_dir / "bad.json").write_text(json.dumps({"id": ["unhashable"], "name": {"k": 1}}), encoding="utf-8")
    (proj_dir / "good.json").write_text(json.dumps({"id": "proj-123", "name": "mptrade"}), encoding="utf-8")
    assert summary_store.resolve_project_ids([" mptrade "], [str(cli_base)]) == {"mptrade", "proj-123"}

    # 3. prepare_topic_session rejects fractional compact_steps
    with pytest.raises(ValueError):
        topics.prepare_topic_session("macro_geopolitical", cli_base=str(cli_base), compact_steps=1.5)  # type: ignore[arg-type]

    # 4. ui_mirror rejects path traversal cids
    assert ui_mirror.sync_topic_thread_from_ui("../bad", bot_cli_base=str(cli_base), ui_cli_base=str(tmp_path / "ui")) is False
    assert ui_mirror.mirror_topic_thread_to_ui("../bad", "Title", bot_cli_base=str(cli_base), ui_cli_base=str(tmp_path / "ui")) is False


def test_batch14_notify_engine_server_alertnotifiers_and_mailer_hygiene(monkeypatch, tmp_path):
    from notify_engine import alertnotifiers, mailer, server

    # 1. server._send_ntfy rejects bool/whitespace topic without posting to root ntfy.sh
    monkeypatch.setenv("NOTIFICATION_QUEUE_FILE", str(tmp_path / "queue.jsonl"))
    monkeypatch.setenv("NOTIFICATION_STATE_FILE", str(tmp_path / "state.json"))
    ns = server.NotificationServer()
    assert ns._send_ntfy("Title", "Body", "default", "   ") is True
    assert ns._send_ntfy("Title", "Body", "default", True) is True  # type: ignore[arg-type]

    # 2. AlertNotifier.send respects string boolean flags ("false")
    sent_calls = []
    monkeypatch.setattr(alertnotifiers.AlertNotifier, "send_email_batch", staticmethod(lambda *a, **kw: sent_calls.append("email")))
    monkeypatch.setattr(alertnotifiers.AlertNotifier, "send_phone_webhook_batch", staticmethod(lambda *a, **kw: sent_calls.append("phone")))
    alertnotifiers.AlertNotifier.send(
        {"type": "bot_event", "name": "Test", "body": "B"},
        enable_console="false",
        enable_file="false",
        enable_email="false",
        enable_phone_webhook="false",
    )
    assert sent_calls == []

    # 3. bind_notify ignores whitespace/bool symbol and whitespace env vars
    monkeypatch.setenv("TEST_SYM_ENV_IT14", "   ")
    captured = []
    monkeypatch.setattr(alertnotifiers, "notify", lambda title, body, src, sym, **kw: captured.append(sym))
    fn = alertnotifiers.bind_notify(("TEST_SYM_ENV_IT14",), "FALLBACK_SYM")
    fn("T", "B", "src", symbol="   ")
    fn("T", "B", "src", symbol=True)  # type: ignore[arg-type]
    assert captured == ["FALLBACK_SYM", "FALLBACK_SYM"]

    # 4. mailer.send_email respects dedup="false" without calling pass_common_gate
    monkeypatch.setenv("DISABLE_EXTERNAL_NOTIFICATIONS", "0")
    monkeypatch.setattr(mailer, "_config", lambda: {"server": "localhost", "port": 25, "username": "u", "password": "p", "to": "t"})
    monkeypatch.setattr(mailer, "_smtp_send", lambda msg, cfg: None)
    monkeypatch.setattr(
        alertnotifiers,
        "pass_common_gate",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("pass_common_gate should not run when dedup='false'")),
    )
    assert mailer.send_email("Subj", "Body", dedup="false") is True  # type: ignore[arg-type]


def test_batch15_pricefetcher_new_coins_and_cooldown_hardening(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from lock.cooldown import Cooldown
    from market_monitor import new_coins_discovery as ncd
    from market_monitor import pricefetcher as pf

    # 1. BinancePricePlatform._load_symbols ignores None/non-string symbol and baseAsset
    class _FakeBinanceResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "symbols": [
                    {"symbol": None, "baseAsset": "BTC", "quoteAsset": "USDC", "status": "TRADING"},
                    {"symbol": "   ", "baseAsset": "ETH", "quoteAsset": "USDC", "status": "TRADING"},
                    {"symbol": " btcusdc ", "baseAsset": None, "quoteAsset": "usdc", "status": "trading"},
                    {"symbol": "ETHUSDC", "baseAsset": " eth ", "quoteAsset": "USDC", "status": "TRADING"},
                ]
            }

    monkeypatch.setattr(pf.requests, "get", lambda *a, **kw: _FakeBinanceResp())
    bp = pf.BinancePricePlatform()
    assert None not in bp._supported_symbols
    assert "" not in bp._supported_symbols
    assert None not in bp._symbol_mapping
    assert bp._symbol_mapping.get("ETH") == "ETHUSDC"
    assert "BTCUSDC" in bp._usdc_pairs

    # 2. PricePlatformFactory.get_price normalizes symbol cache keys and rejects blank symbols
    pf_fac = object.__new__(pf.PricePlatformFactory)

    class _DummyPlat:
        platform_name = "Dummy"

        def supports_symbol(self, sym):
            return sym.strip().upper() == "BTC"

        def get_price(self, sym):
            return 50000.0

    pf_fac._platforms = [_DummyPlat()]
    pf_fac._symbol_platform_cache = {}
    res = pf_fac.get_price(" btc ")
    assert res["price"] == 50000.0
    assert pf_fac._symbol_platform_cache == {"BTC": "Dummy"}
    with pytest.raises(ValueError):
        pf_fac.get_price("   ")

    # 3. new_coins_discovery handles aware/string datetimes, string bools, and empty alert lists
    dex = object.__new__(ncd.DexScreenerSource)
    dex._cache = [
        {"symbol": "FOO", "added_at": datetime.now(timezone.utc)},
        {"symbol": "BAR", "added_at": "invalid-date"},
    ]
    assert len(dex.get_new_coins(3)) == 1

    cmc = object.__new__(ncd.CoinMarketCapSource)
    cmc._cache = [
        {"symbol": "BAZ", "added_at": datetime.now(timezone.utc).isoformat()},
    ]
    assert len(cmc.get_new_coins(3)) == 1

    mon = object.__new__(ncd.NewCoinsMonitor)
    mon.price_monitor = None
    mon.all_new_coins = {}
    mon.all_symbols = set()
    cb_calls = []
    mon.alert_callbacks = [lambda alerts: cb_calls.append(alerts)]
    mon._trigger_alerts([{"symbol": "   "}, None], "coinmarketcap", auto_add="false")  # type: ignore[list-item,arg-type]
    assert cb_calls == []

    # 4. Cooldown group methods reject blank/bool group_id/member, handle keep_group="false", and survive non-list fields
    cd = Cooldown(
        "test_b15",
        state_path=str(tmp_path / "cd.json"),
        lock_path=str(tmp_path / "cd.lock"),
    )
    assert cd.reserve_group_member("BTCUSDC", 60, None, "BUY")[0] is False
    assert cd.reserve_group_member("BTCUSDC", 60, True, "BUY")[0] is False
    assert cd.reserve_group_member("BTCUSDC", 60, "grp1", "   ")[0] is False

    ok, entry = cd.reserve_group_member("BTCUSDC", 60, "grp1", "BUY", group_members=["spoofed"], timestamp=0)
    assert ok is True
    assert entry["group_id"] == "grp1"
    assert entry["group_members"] == ["BUY"]
    assert entry["timestamp"] > 0

    # Corrupt group fields on disk and verify commit & release recover cleanly
    state = cd._read()
    state["BTCUSDC"]["group_pending"] = 123
    state["BTCUSDC"]["group_committed"] = 456
    state["BTCUSDC"]["group_results"] = "not-a-dict"
    cd._write(state)
    cd.commit_group_member("BTCUSDC", "grp1", "BUY", order_id=99)
    assert cd.get("BTCUSDC")["group_committed"] == ["BUY"]
    assert cd.get("BTCUSDC")["group_results"] == {"BUY": {"order_id": 99}}
    assert cd.release_group_member("BTCUSDC", "grp1", "BUY", keep_group="false") is True
    assert cd.get("BTCUSDC") is None


def test_batch16_orchestrator_os_watchdogs_and_telemetry_daemons_hardening(monkeypatch, tmp_path):
    import math
    from intelligence import globaltelemetry_collector as gtc
    from intelligence import macro_analyzer as ma
    from orchestratorOS.admin import agy_store_fix as asf
    from orchestratorOS.admin import prune_llm_threads as plt
    from orchestratorOS.lib import watchdog_common as wc
    from orchestratorOS.livecheck import os_resources_watchdog as orw

    # 1. prune_llm_threads._as_bool rejects NaN and prune_old_cli_threads blocks whitespace-padded protected project_id
    assert plt._as_bool(float("nan"), default=False) is False
    assert plt._as_bool("garbage", default=True) is True

    db_path = tmp_path / "conversation_summaries.db"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE conversation_summaries (conversation_id TEXT, project_id TEXT, title TEXT, preview TEXT, step_count TEXT, last_modified_time TEXT)"
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES ('c_human', 'default-cli-project', 'User Chat', 'Hello', '10.0', '2020-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    assert plt.prune_old_cli_threads(cli_base=str(tmp_path), project_id="  mptrade  ", dry_run=True) == (0, 0)
    # Float-string step_count ("10.0" > 6) must be preserved as human interactive thread even with all_threads_in_project=True
    assert plt.prune_old_cli_threads(cli_base=str(tmp_path), all_threads_in_project=True, dry_run=True) == (0, 0)

    # 2. watchdog_common._valid_webhook_url rejects bare slash topic or root ntfy URL without topic
    assert wc._valid_webhook_url("/") is None
    assert wc._valid_webhook_url("///") is None
    assert wc._valid_webhook_url("https://ntfy.sh/") is None
    assert wc._valid_webhook_url("/my-topic") == "https://ntfy.sh/my-topic"
    assert wc._valid_webhook_url("https://ntfy.sh/my-topic") == "https://ntfy.sh/my-topic"

    # 3. os_resources_watchdog._as_bool_flag normalizes string "false" in persisted state
    assert orw._as_bool_flag("false", default=True) is False
    assert orw._as_bool_flag(float("nan"), default=False) is False

    # 4. GlobalTelemetryCollector and MacroAnalyzer write_heartbeat sanitize non-dict status and NaN ts
    col = object.__new__(gtc.GlobalTelemetryCollector)
    col.symbols = ["BTCUSDC"]
    col.orderbook_interval_sec = 15.0
    col.derivatives_interval_sec = 30.0
    col.whale_interval_sec = 60.0
    col.news_interval_sec = 120.0
    col.fear_greed_interval_sec = 1800.0
    col.breadth_interval_sec = 900.0
    col.heartbeat_path = str(tmp_path / "gtc.hb")
    col.legacy_heartbeat_path = str(tmp_path / "gtc_legacy.hb")
    col.write_heartbeat({"ts": float("nan")})
    hb_data = json.loads((tmp_path / "gtc.hb").read_text(encoding="utf-8"))
    assert math.isfinite(hb_data["ts"]) and hb_data["ts"] > 0

    ana = object.__new__(ma.MacroAnalyzer)
    ana.macro_llm_interval_sec = 3600.0
    ana.advisor_interval_sec = 1800.0
    ana.thread_prune_interval_sec = 10800.0
    ana.heartbeat_path = str(tmp_path / "ma.hb")
    ana.write_heartbeat(None)  # type: ignore[arg-type]
    ma_hb = json.loads((tmp_path / "ma.hb").read_text(encoding="utf-8"))
    assert math.isfinite(ma_hb["ts"]) and ma_hb["ts"] > 0
    assert asf._as_bool_flag("false", True) is False


def test_batch17_intelligence_daemon_gemini_client_advisor_and_cli_hardening(capsys, tmp_path):
    import math
    from intelligence import cli as icli
    from intelligence import daemon as idaemon
    from intelligence.sentiment.gemini_client import GeminiClient
    from intelligence.sentiment.sentiment_advisor import SentimentAdvisor, SentimentAdvisorAssessment

    # 1. IntelligenceTelemetryDaemon._as_bool rejects NaN and write_heartbeat sanitizes NaN/None status
    assert idaemon._as_bool(float("nan")) is False
    dm = object.__new__(idaemon.IntelligenceTelemetryDaemon)
    dm.symbols = ["BTCUSDC"]
    dm.orderbook_interval_sec = 15.0
    dm.derivatives_interval_sec = 30.0
    dm.whale_interval_sec = 60.0
    dm.macro_interval_sec = 120.0
    dm.macro_llm_interval_sec = 3600.0
    dm.heartbeat_path = str(tmp_path / "dm.hb")
    dm.write_heartbeat({"ts": float("nan"), "extra": float("inf")})
    dm_hb = json.loads((tmp_path / "dm.hb").read_text(encoding="utf-8"))
    assert math.isfinite(dm_hb["ts"]) and dm_hb["ts"] > 0
    assert dm_hb["last_run"]["extra"] is None

    # 2. GeminiClient rejects blank/non-string prompts without calling custom_runner and sanitizes blank model/api_key
    calls = []
    gc = GeminiClient(model="   ", api_key="   ", project_id="   ", custom_runner=lambda p, m, t: (calls.append((p, m, t)), "ok")[1])
    assert gc.api_key is None
    assert gc.model != ""
    assert gc.project_id != ""
    assert gc.query_text("   ") is None
    assert gc.query_text(None) is None  # type: ignore[arg-type]
    assert calls == []
    assert gc.query_text("valid prompt", model="   ") == "ok"
    assert calls[0][1] == gc.model

    # 3. SentimentAdvisor.review treats NaN force_refresh as False so fresh cached assessment is returned
    adv = object.__new__(SentimentAdvisor)
    adv.cache_ttl_sec = 3600.0
    adv._last_eval_ts = 1000.0
    cached = SentimentAdvisorAssessment("NEUTRAL", "MODERATE", 0.5, "ok", [], "HOLD", 1000.0)
    adv._cached_assessment = cached
    assert adv.review(force_refresh=float("nan"), now_ts=1050.0) is cached  # type: ignore[arg-type]

    # 4. intelligence.cli.print_intelligence_report handles non-dict report and string "false" allowed flag
    icli.print_intelligence_report(None)  # type: ignore[arg-type]
    icli.print_intelligence_report(
        {
            "symbol": "BTCUSDC",
            "guard_preview": {
                "buy_standard": {"allowed": "false", "suggested_scale": 0.0, "reason": "vetoed"},
            },
        }
    )
    out = capsys.readouterr().out
    assert "⛔ BLOCKED" in out


def test_batch18_geopolitical_news_and_sentiment_collectors_guards_hardening(tmp_path):
    from types import SimpleNamespace
    from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer, GeopoliticalThreatAssessment
    from intelligence.macro.news_feed_collector import NewsFeedCollector, NewsHeadline
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector, FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
    from intelligence.sentiment.guards.high_stake_guard import evaluate_llm_guard_response, is_llm_guard_exempt
    from intelligence.sentiment.guards.post_stop_reentry_guard import PostStopReentryGuard, get_post_stop_reentry_guard

    # 1. GeopoliticalThreatAnalyzer._as_bool_flag rejects NaN and string "false" for shock/high-severity flags
    ga = object.__new__(GeopoliticalThreatAnalyzer)
    ga.cache_ttl_sec = 3600.0
    ga.fallback_retry_ttl_sec = 900.0
    ga._last_eval_ts = 1000.0
    ga._last_headlines_digest = ""
    cached_ga = GeopoliticalThreatAssessment("NORMAL", 0.1, "ok", "NONE", 1, 1000.0, is_fallback=False)
    ga._cached_assessment = cached_ga
    fake_snap = SimpleNamespace(
        has_critical_shock_keywords="false",
        headlines=[SimpleNamespace(title="Calm news", is_high_severity="false")],
    )
    assert ga.assess(fake_snap, force_refresh=float("nan"), now_ts=1050.0) is cached_ga  # type: ignore[arg-type]

    # 2. NewsFeedCollector._aggregate & get_pool_snapshot handle string "false" is_high_severity and non-snapshot input
    nfc = object.__new__(NewsFeedCollector)
    nfc.rolling_window_sec = 3600.0
    nfc.history_file = str(tmp_path / "news.json")
    nfc._aggregated_pool = {
        "calm": (
            1000.0,
            NewsHeadline("Calm headline", "Src", "", "", (), is_high_severity="false"),  # type: ignore[arg-type]
        )
    }
    snap = nfc.get_pool_snapshot(now_ts=1010.0)
    assert snap.high_severity_count == 0
    agg = nfc._aggregate(None, 1010.0)  # type: ignore[arg-type]
    assert agg.high_severity_count == 0

    # 3. FearGreedCollector.fetch & MarketBreadthCollector.parse_tickers / fetch handle NaN force_refresh and tuple tickers
    fgc = FearGreedCollector(cache_ttl_sec=3600.0)
    fg_snap = FearGreedSnapshot(50, "Neutral", 1000, (50,), 0, False, False)
    fgc._cached_snapshot = fg_snap
    fgc._last_fetch_ts = 1000.0
    mbc = MarketBreadthCollector(cache_ttl_sec=3600.0, min_volume_usd=100.0)
    tickers_tuple = tuple(
        {"symbol": f"COIN{i}USDT", "quoteVolume": "1000", "priceChangePercent": "2.5"}
        for i in range(6)
    )
    mb_snap = mbc.parse_tickers(tickers_tuple, min_volume_usd=100.0, now_ts=1000.0)  # type: ignore[arg-type]
    assert mb_snap is not None and mb_snap.total_count == 6

    # 4. PostStopReentryGuard mode/notify normalization and HighStakeGuard helpers
    psg = PostStopReentryGuard(mode=0, notify_enabled=float("nan"))  # type: ignore[arg-type]
    assert psg.mode == "off"
    assert psg.notify_enabled is False
    psg2 = get_post_stop_reentry_guard(
        margins={
            "post_stop_reentry_mode": "enforce",
            "intelligence_guards_mode": False,
            "llm_guard_mode": "enforce",
            "post_stop_reentry_timeout_sec": 10.0,
            "post_stop_reentry_fallback": "allow",
            "post_stop_reentry_notify": float("nan"),
            "post_stop_reentry_cache_ttl_sec": 60.0,
        }
    )
    assert psg2.mode == "off"
    assert psg2.notify_enabled is False
    dec = evaluate_llm_guard_response("HighStakeGuard", "BUY", None, fallback_action="allow", notional_eur=float("nan"))
    assert dec.metadata["notional_eur"] == 0.0
    assert is_llm_guard_exempt("true") is False


def test_batch19_external_and_internal_collectors_guards_triggers_hardening(tmp_path):
    from types import SimpleNamespace
    from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
    from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
    from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
    from intelligence.external.guards.smart_stop_loss_guard import SmartStopLossGuard
    from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger
    from intelligence.internal.guards.trend_significance_guard import TrendSignificanceGuard
    from intelligence.internal.state.survival import get_trend_survival_metrics
    from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent

    # 1. External collectors reject allow_network="false" / NaN without network calls and ignore None in record_*
    dt_col = DerivativesTelemetryCollector(cache_dir=str(tmp_path))
    ob_col = OrderbookDepthCollector(cache_dir=str(tmp_path))
    wh_col = WhalePositioningCollector(cache_dir=str(tmp_path))
    assert dt_col.fetch("BTCUSDC", allow_network="false") is None  # type: ignore[arg-type]
    assert ob_col.fetch("BTCUSDC", allow_network=float("nan")) is None  # type: ignore[arg-type]
    assert wh_col.fetch("BTCUSDC", allow_network="0") is None  # type: ignore[arg-type]
    dt_col.record_telemetry(None)  # type: ignore[arg-type]
    ob_col.record_snapshot(None)  # type: ignore[arg-type]
    wh_col.record_snapshot(None)  # type: ignore[arg-type]

    # 2. WhalePositioningCollector.classify_regime normalizes prev_regime casing/whitespace
    assert (
        WhalePositioningCollector.classify_regime(
            price_1h_change_pct=0.5,
            oi_change_pct=0.5,
            taker_ratio=1.20,
            prev_regime="  AGGRESSIVE_SHORTING  ",
        )
        == "short_squeeze_primed"
    )

    # 3. SmartStopLossGuard normalizes numeric/bool modes, symbol separators, and uppercase divergence_regime
    ssl_off = SmartStopLossGuard(mode=0)  # type: ignore[arg-type]
    assert ssl_off.mode == "off"
    ssl_enf = SmartStopLossGuard(mode="enforce")
    should_exec, reason, meta = ssl_enf.evaluate(
        "BTC-USDC",
        0.035,
        0.03,
        provider="sim",
        whale_snapshot=SimpleNamespace(
            top_traders_long_pct=0.30,
            taker_buy_sell_ratio=0.60,
            divergence_regime="  AGGRESSIVE_SHORTING  ",
        ),
    )
    assert should_exec is True
    assert "active_crash_cascade" in reason
    assert meta["symbol"] == "BTCUSDC"

    # 4. WhaleAccumulationTrigger handles snapshot without divergence_regime attribute
    wat = WhaleAccumulationTrigger()
    ev = wat.evaluate(
        "BTCUSDC",
        60000.0,
        snapshot=SimpleNamespace(top_traders_long_pct=0.70, taker_buy_sell_ratio=1.50, open_interest_1h_change_pct=2.0),  # type: ignore[arg-type]
        now=1000.0,
    )
    assert ev is not None and ev.metadata["regime"] == "neutral"

    # 5. TriggerEvent.is_executable and TrendSignificanceGuard normalize short side "B"/"S", and survival rejects NaN allow_network
    te = TriggerEvent(action=TriggerAction.ENTRY, side="B", source="test", symbol="BTCUSDC", price=100.0, strength=1.0)  # type: ignore[arg-type]
    assert te.is_executable() is True
    tsg = TrendSignificanceGuard()
    dec = tsg.evaluate(te, [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0])
    assert dec.allowed is False and "trend_direction_inverted" in dec.reason
    surv = get_trend_survival_metrics("NONEXISTENT_SYM_XYZ", 3600.0, allow_network=float("nan"))  # type: ignore[arg-type]
    assert surv["symbol"] == "NONEXISTENT_SYM_XYZ"


def test_batch20_visuals_summary_store_and_composite_hardening(tmp_path):
    import numpy as np
    from intelligence.composite import CompositeMarketIntelligence
    from orchestratorOS.agy import summary_store as ss
    from visuals import drawprices, tradeall_observe

    # 1. tradeall_observe.format_state_text handles string/float trend labels
    txt = tradeall_observe.format_state_text(
        {"current_price": 65000, "final_trend": "UP", "trend_fast": "-1.0", "kalman_trend": "HOLD"},
        "HEADER",
    )
    assert "trend:     UP" in txt
    assert "fast:      DOWN" in txt
    assert "kalman:    FLAT" in txt

    # 2. drawprices.min_max_scale and make_sequences tolerate None/bad-string/float-string inputs
    scaled_none, vmin_n, vmax_n = drawprices.min_max_scale(None)  # type: ignore[arg-type]
    assert len(scaled_none) == 0 and vmin_n == 0.0 and vmax_n == 0.0
    scaled, vmin, vmax = drawprices.min_max_scale([10.0, "bad_str", "20.0", 30.0])  # type: ignore[arg-type]
    assert len(scaled) == 3 and vmin == 10.0 and vmax == 30.0
    X, y = drawprices.make_sequences(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "2.0")  # type: ignore[arg-type]
    assert X.shape == (3, 2, 1) and len(y) == 3

    # 3. summary_store.drop_wal_sidecars rejects force="false" / NaN, and write_annotation_title sanitizes newlines/None
    db_file = tmp_path / "test.db"
    wal_file = tmp_path / "test.db-wal"
    db_file.write_bytes(b"db")
    wal_file.write_bytes(b"non-empty-wal")
    ss.drop_wal_sidecars(str(db_file), force="false")  # type: ignore[arg-type]
    assert wal_file.exists()
    ss.drop_wal_sidecars(str(db_file), force=float("nan"))  # type: ignore[arg-type]
    assert wal_file.exists()
    ss.write_annotation_title(str(tmp_path), "  cid123  ", "Line1\nLine2")
    assert ss.read_annotation_title(str(tmp_path), "cid123") == "Line1 Line2"
    ss.write_annotation_title(str(tmp_path), "../bad", "Title")
    assert not (tmp_path / "annotations" / "bad.pbtxt").exists()

    # 4. CompositeMarketIntelligence.evaluate normalizes string/NaN price into finite float
    cmi = CompositeMarketIntelligence()
    ev1 = cmi.evaluate("BTCUSDC", "64500.5", ts=1000.0)  # type: ignore[arg-type]
    assert ev1.price == 64500.5
    ev2 = cmi.evaluate("BTCUSDC", float("nan"), ts=1001.0)
    assert ev2.price == 0.0 and ev2.can_execute is False


def test_batch21_monitortrades_position_exit_manager_active_intents_and_archiver_hardening():
    from types import SimpleNamespace
    import active_intents
    import monitortrades as mt
    import tradeall_price_archiver as archiver
    from strategies.position_exit_manager import (
        PositionExitConfig,
        PositionExitManager,
        PositionSnapshot,
        ReentrySnapshot,
    )

    # 1. monitortrades._FifoTradeIdTracker sanitizes maxsize and refreshes FIFO order
    tracker = mt._FifoTradeIdTracker(maxsize=True)  # type: ignore[arg-type]
    assert tracker._maxsize == 10000
    small_tracker = mt._FifoTradeIdTracker(maxsize=4)
    for k in ("t1", "t2", "t3", "t4"):
        small_tracker.add(k)
    small_tracker.add("t1")  # refresh t1 to most recent
    small_tracker.add("t5")  # prune oldest 50% (t2, t3, t4) -> keeps t1, t5
    assert "t1" in small_tracker and "t5" in small_tracker

    # 2. monitortrades.get_relevant_trade handles tuples and get_position_stats handles None from api.get_orders
    orders_tuple = (
        {"price": 60000.0, "qty": 0.1, "time": 1700000000000},
        {"price": 62000.0, "qty": 0.1, "time": 1700000060000},
    )
    px, ts_s, ok = mt.get_relevant_trade(orders_tuple, "BUY", symbol="BTCUSDC")
    assert ok is True and px == 62000.0 and ts_s == 1700000060.0
    none_api = SimpleNamespace(name="none_api", get_orders=lambda sym, side, maxage: None)
    stats = mt.get_position_stats("BTC-USDC", 3600, api=none_api, force_refresh=True)
    assert stats["buy_count"] == 0 and stats["sell_count"] == 0 and stats["net_qty"] == 0.0

    # 3. PositionSnapshot and ReentrySnapshot normalize string/bool/NaN numeric fields, and scalar vol_resolver works
    ps = PositionSnapshot(
        symbol="BTCUSDC",
        current_price="60000",  # type: ignore[arg-type]
        avg_buy_price="64000",  # type: ignore[arg-type]
        net_qty="1.5",  # type: ignore[arg-type]
        sellable_qty=True,  # type: ignore[arg-type]
        current_time_s="1700000000",  # type: ignore[arg-type]
        vol_1h=float("nan"),
        sma_value="61000.5",  # type: ignore[arg-type]
    )
    assert ps.net_qty == 1.5 and ps.sellable_qty == 0.0 and ps.current_time_s == 1700000000.0 and ps.vol_1h == 0.0
    assert ps.sma_value == 61000.5
    rs = ReentrySnapshot(
        symbol="BTCUSDC",
        current_price="60000",  # type: ignore[arg-type]
        last_sell_price="62000",  # type: ignore[arg-type]
        current_time_s=True,  # type: ignore[arg-type]
        last_sell_time_s="1699999000",  # type: ignore[arg-type]
        vol_1h="0.025",  # type: ignore[arg-type]
    )
    assert rs.current_time_s == 0.0 and rs.last_sell_time_s == 1699999000.0 and rs.vol_1h == 0.025

    mgr = PositionExitManager(config=PositionExitConfig(allow_external_cache=False), vol_resolver=lambda _s: 0.03)  # type: ignore[arg-type]
    buf, meta = mgr.compute_disaster_buffer("BTCUSDC", None)  # type: ignore[arg-type]
    assert buf > 0.0 and meta["vol_1h"] == 0.03
    pb, pb_meta = mgr.compute_tp_anomaly_pullback("BTCUSDC", None, "0.25")  # type: ignore[arg-type]
    assert pb > 0.0 and pb_meta["vol_1h"] == 0.03

    # 4. active_intents._intent_row normalizes shorthand sides ("B"/"S") and string terminal_status
    row_b = active_intents._intent_row(
        {"intent_id": "id-b", "side": "b", "qty": 1.0, "symbol": "BTCUSDC"},
        owner="kraken",
        source="s",
        location="$",
        inherited={},
    )
    assert row_b is not None and row_b["side"] == "BUY"
    row_term = active_intents._intent_row(
        {"intent_id": "id-term", "side": "SELL", "qty": 1.0, "status": "accepted", "terminal_status": "filled"},
        owner="kraken",
        source="s",
        location="$",
        inherited={},
    )
    assert row_term is None

    # 5. tradeall_price_archiver._shutdown isolates component failures so remaining caches flush and ws stops
    events = []
    bad_cache = SimpleNamespace(
        shutdown=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        save_state_to_file=lambda: events.append("bad_flushed"),
    )
    good_cache = SimpleNamespace(
        shutdown=lambda: events.append("good_shutdown"),
        save_state_to_file=lambda: events.append("good_flushed"),
    )
    cp_mgr = SimpleNamespace(
        unsubscribe_price=lambda c: events.append("unsub"),
        shutdown=lambda: events.append("cp_shutdown"),
    )
    ws_mod = SimpleNamespace(bapi_ws_manager=SimpleNamespace(stop=lambda: events.append("ws_stop")))
    archiver._shutdown([bad_cache, good_cache], cp_mgr, ws_mod)
    assert events == ["unsub", "unsub", "bad_flushed", "good_shutdown", "good_flushed", "cp_shutdown", "ws_stop"]


def test_batch22_binance_ws_market_data_trailing_stop_and_providers_hardening():
    import asyncio
    from types import SimpleNamespace
    import binance_api.bapi_ws as bapi_ws
    import binance_api.bapi_ws_bridge as bapi_ws_bridge
    import binance_api.market_data as b_md
    from binance_api.trailing_stop import TrailingStop
    from providers.base import MarketDataProvider, aggregate_fills_by_order_id, extract_order_side
    from providers.quantity import balance_cap_quantity
    from providers.replay_provider import ReplayMarketDataProvider

    # 1. bapi_ws._as_bool, _sanitize_timeout, and _interruptible_sleep handle bool/NaN/negative/0 inputs safely
    assert bapi_ws._as_bool("false", default=True) is False
    assert bapi_ws._as_bool(float("nan"), default=False) is False
    assert bapi_ws._sanitize_timeout(True, default=8.0) == 8.0
    assert bapi_ws._sanitize_timeout(-5.0, default=8.0) == 8.0
    assert bapi_ws._sanitize_timeout("2.5", default=8.0) == 2.5

    ws_base = bapi_ws.BinanceWSBase()
    asyncio.run(ws_base._interruptible_sleep(float("nan"), step=0.0))
    asyncio.run(ws_base._interruptible_sleep(-1.0, step=-1.0))

    # 2. bapi_ws_bridge.refresh_symbol_in_cache does not raise AttributeError when manager lacks cls_name
    bapi_ws_bridge.refresh_symbol_in_cache(SimpleNamespace(), "BTCUSDC")

    # 3. binance_api.market_data supports 43200 ("1M"), whitespace-padded and uppercase intervals (" 1M ", "1H", "1D", "1W")
    assert b_md.interval_to_binance_str(43200) == "1M"
    assert b_md.interval_to_binance_str(" 1M ") == "1M"
    assert b_md.interval_to_binance_str(" 1H ") == "1h"
    assert b_md.interval_to_binance_str("1D") == "1d"
    assert b_md.interval_to_binance_str("1W") == "1w"
    assert b_md.parse_interval_minutes(" 4H ") == 240
    assert b_md._parse_bool_flag("false", default=True) is False
    assert b_md._parse_bool_flag(float("nan"), default=True) is True

    # 4. TrailingStop._safe_float and logging helpers tolerate string/None/bool prices and peaks
    assert TrailingStop._safe_float("123.45", 0.0) == 123.45
    assert TrailingStop._safe_float(True, 5.0) == 5.0
    assert TrailingStop._safe_float(float("nan"), 5.0) == 5.0
    dummy_ts = object.__new__(TrailingStop)
    logged = []
    dummy_ts.log = logged.append
    dummy_ts.log_dry_sell("BTC", "BTC", "BTCUSDC", 0.1, "60000", None, 2.5)
    dummy_ts.log_dry_rebuy("BTC", "BTC", "BTCUSDC", 0.1, "58000", {"low": "invalid"})
    dummy_ts.log_hold("BTC", "BTC", "BTCUSDC", "60000", "61000", None, 2.5, 0.1)
    assert len(logged) == 3

    # 5. providers/base, quantity, and replay_provider normalize BID/ASK/LONG/SHORT sides and None ohlc_series
    assert extract_order_side({"side": "bid"}) == "BUY"
    assert extract_order_side({"side": "short"}) == "SELL"
    assert extract_order_side({"type": "ask"}) == "SELL"
    agg = aggregate_fills_by_order_id(
        [{"orderId": "10", "side": "BID", "price": 100.0, "qty": 2.0, "timestamp": 1000}],
        side="long",
    )
    assert len(agg) == 1 and agg[0]["side"] == "BUY"
    cap, asset = balance_cap_quantity(lambda _a: 500.0, "BTCUSDC", "bid", 100.0)
    assert cap == 5.0 and asset == "USDC"

    class _DummyProvider(MarketDataProvider):
        @property
        def name(self):
            return "Dummy"

        def get_current_price(self, symbol: str):
            return 100.0

        def supports_symbol(self, symbol: str) -> bool:
            return True

        def ohlc_series(self, symbol: str, interval_min: int):
            return None

    dp = _DummyProvider()
    assert dp.ohlc_closes("BTCUSDC", 60) == []

    rp = ReplayMarketDataProvider({"BTCUSDC": [(1000.0, 50000.0)]})
    rp._series["ETHUSDC"] = [(1000.0, 3000.0)]
    assert rp.advance("ETHUSDC", 1) == 3000.0
    assert rp.place_order("ETHUSDC", "bid", 3000.0, 1.0) is not None
    assert len(rp.get_orders("ETHUSDC", "long", 60.0)) == 1


def test_batch23_market_monitor_lock_and_notify_engine_hardening(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import market_monitor.alerts_config as alerts_config
    from market_monitor.pricechecker import PriceChecker
    from market_monitor.pricefetcher import CacheAllPriceFetcherManager
    from lock.file_lock import FileLock
    import lock.trade_cooldown as tc
    from notify_engine.server import NotificationServer

    # 1. alerts_config.load_config rejects boolean/non-path inputs (preventing os.path.exists(True) fd=1 bug)
    with pytest.raises(ValueError):
        alerts_config.load_config(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        alerts_config.load_config("   ")

    # alerts_config.resolve handles NaN is_dynamic as False
    ac = {"default": {"up_percent": 4.0, "down_percent": 7.0}, "dynamic": {"up_percent": 12.0, "down_percent": 25.0}}
    assert alerts_config.resolve(ac, "BTC", float("nan")) == {"up_percent": 4.0, "down_percent": 7.0}

    # 2. PriceChecker handles None original_symbols/symbols and case-insensitive _is_dynamic_symbol
    cache_stub = SimpleNamespace(original_symbols=None, symbols=None, symbol_added_time={"BTC": 1700000000.0})
    pc = PriceChecker(cache_stub)
    assert pc._is_dynamic_symbol(" btc ") is True
    assert pc.check_all_symbols() == []
    assert pc.get_status()["symbols_count"] == 0

    # 3. CacheAllPriceFetcherManager strips symbols in remove_symbol/get_latest_price/get_price_history and accepts float/str integer limits
    pf = object.__new__(CacheAllPriceFetcherManager)
    pf.lock = threading.Lock()
    pf.active_symbols = {"BTC"}
    pf.symbols = ["BTC"]
    pf.original_symbols = ["BTC"]
    pf.symbol_added_time = {"BTC": 1700000000.0}
    pf.symbol_preferred_source = {"BTC": "Binance"}
    pf.cache = {"BTC": [[1700000000000, 62500.0]]}
    assert pf.get_latest_price(" BTC ") == 62500.0
    hist = pf.get_price_history(" BTC ", limit="10")  # type: ignore[arg-type]
    assert len(hist) == 1 and hist[0]["price"] == 62500.0
    assert pf.remove_symbol(" BTC ") is True
    assert "BTC" not in pf.active_symbols

    # 4. FileLock auto-creates parent directories and trade_cooldown normalizes shorthand sides ("b"/"BUY")
    nested_lock = tmp_path / "nested" / "sub" / "test.lock"
    with FileLock(nested_lock):
        assert nested_lock.exists()

    monkeypatch.setattr(tc, "STATE_FILE", str(tmp_path / "tc.json"))
    monkeypatch.setattr(tc, "LOCK_FILE", str(tmp_path / "tc.lock"))
    tc._cd = None
    with tc.trade_slot("b", "BTCUSDC", cooldown_sec=60, pair_id="pair-1") as s1:
        assert s1.allowed is True
        with tc.trade_slot("BUY", "BTCUSDC", cooldown_sec=60, pair_id="pair-1") as s2:
            assert s2.allowed is False
        with tc.trade_slot("s", "BTCUSDC", cooldown_sec=60, pair_id="pair-1") as s3:
            assert s3.allowed is True
            s3.commit("999")
        s1.commit("888")
    assert tc.release_pair_leg("BTCUSDC", "pair-1", "buy") is True

    # 5. NotificationServer.process_line tolerates non-str line and None/bool bot_name
    srv = object.__new__(NotificationServer)
    srv.traceback_buffers = {}
    srv.rules = []
    srv.cooldowns = {}
    srv.process_line(None, None)  # type: ignore[arg-type]
    srv.process_line(True, False)  # type: ignore[arg-type]


def test_batch24_execution_evaluation_and_financial_benchmark_hardening():
    import argparse
    from offline.backtests.execution import ExecutionModel, FeeModel, split_order_fill
    from offline.backtests.evaluation import (
        assess_decision_evidence,
        build_execution_model_from_args,
    )
    from offline.backtests.financial_benchmark import aggregate_financial_windows

    # 1. FeeModel / ExecutionModel reject None/non-numeric and parse string market flag safely
    with pytest.raises(ValueError, match="limit_fee_pct"):
        FeeModel(limit_fee_pct=None, market_fee_pct=0.1)  # type: ignore[arg-type]
    fee = FeeModel(limit_fee_pct="0.1", market_fee_pct="0.2")  # type: ignore[arg-type]
    assert fee.rate_pct(market="false") == pytest.approx(0.1)
    assert fee.rate_pct(market="true") == pytest.approx(0.2)

    exec_model = ExecutionModel(spread_bps="2.0", market_slippage_bps="5.0", partial_fill_ratio="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="high, low, and limit"):
        exec_model.limit_touched("buy", high=105.0, low=False, limit=100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="high, low, and limit"):
        exec_model.limit_touched("sell", high=True, low=95.0, limit=100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="open_price"):
        exec_model.market_price("buy", True)  # type: ignore[arg-type]

    # 2. split_order_fill rejects NaN/bool/None quantities and handles None/NaN amount safely
    with pytest.raises(ValueError, match="the order has no quantity left to execute"):
        split_order_fill({"quantity": float("nan")}, quantity_key="quantity", ratio=0.5)
    with pytest.raises(ValueError, match="the order has no quantity left to execute"):
        split_order_fill({"quantity": True}, quantity_key="quantity", ratio=0.5)
    order_dict = {"quantity": 10.0, "amount": None, "side": None, "_replay_original_quantity": float("nan"), "_replay_had_fill": True}
    filled, chunk, complete = split_order_fill(
        order_dict,
        quantity_key="quantity",
        ratio=0.5,
        amount_key="amount",
    )
    assert filled["quantity"] == pytest.approx(5.0)
    assert chunk == pytest.approx(5.0)
    assert complete is False
    assert order_dict["quantity"] == pytest.approx(5.0)

    # 3. assess_decision_evidence tolerates bool/NaN timestamps and fold_count/total_cycles
    with pytest.raises(ValueError, match="invalid record timestamps"):
        assess_decision_evidence([{"timestamp": True}, {"timestamp": 2000.0}], {})
    ev = assess_decision_evidence(
        [{"timestamp": 1000.0}, {"timestamp": 1000.0 + 40 * 86400.0}],
        {"fold_count": True, "total_cycles": float("nan"), "worst_return_pct": float("nan")},
    )
    assert ev["history_days"] == pytest.approx(40.0)
    assert "folds 0 < 20" in ev["data_issues"]
    assert "completed_cycles 0 < 3" in ev["data_issues"]

    # 4. build_execution_model_from_args rejects boolean window args
    parser = argparse.ArgumentParser()
    bad_args = argparse.Namespace(
        spread_bps=0.0,
        market_slippage_bps=0.0,
        partial_fill_ratio=1.0,
        intrabar_policy="buy_first",
        train=True,
        validation=10,
        test=10,
        step=5,
        warmup=0,
    )
    with pytest.raises(SystemExit):
        build_execution_model_from_args(parser, bad_args)

    # 5. aggregate_financial_windows validates initial_capital and regime_threshold_pct
    with pytest.raises(ValueError, match="initial_capital"):
        aggregate_financial_windows([], initial_capital=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="regime_threshold_pct"):
        aggregate_financial_windows([], initial_capital=1000.0, regime_threshold_pct="bad")  # type: ignore[arg-type]


def test_batch25_datasets_rtrade_pair_exit_manager_and_verify_tools_hardening():
    from types import SimpleNamespace
    from offline.backtests.datasets import align_previous_values, drop_incomplete_last_bar, normalize_record
    from strategies.spot_replay import align_spot_interval_params
    from strategies.rtrade_pair import PairPolicy, anchored_exit_price, quote_prices
    from strategies.position_exit_manager import PositionExitConfig, PositionExitManager, PositionSnapshot
    import verify_tools.portfolio_snapshot as ps

    # 1. datasets: normalize_record / drop_incomplete_last_bar / align_previous_values
    with pytest.raises(ValueError, match="missing required field"):
        normalize_record({"timestamp": 1000, "open": 1, "high": 2, "low": 1})
    with pytest.raises(ValueError, match="cannot be None or boolean"):
        normalize_record({"timestamp": 1000, "open": None, "high": 2, "low": 1, "close": 1.5})
    with pytest.raises(ValueError, match="now_timestamp"):
        drop_incomplete_last_bar([], interval_minutes=60, now_timestamp=True)  # type: ignore[arg-type]
    aligned = align_previous_values(
        [1000, 2000],
        [{"timestamp": "1000.0", "close": "4.5"}, {"timestamp": 1000, "close": "5.0"}],
    )
    assert aligned == [5.0, 5.0]

    # 2. spot_replay: align_spot_interval_params supports SimpleNamespace and sanitizes bool regime_min_samples
    ns_params = SimpleNamespace(
        trend_interval=60,
        tp_trail_vol_interval=60,
        dca_vol_interval=60,
        regime_min_samples=True,
    )
    aligned_p = align_spot_interval_params(ns_params, 240)
    assert aligned_p.trend_interval == 240
    assert isinstance(aligned_p.regime_min_samples, int) and not isinstance(aligned_p.regime_min_samples, bool)

    # 3. rtrade_pair: PairPolicy, quote_prices, anchored_exit_price accept numeric strings and shorthand sides
    pol = PairPolicy(adjustment_fraction="0.01", price_decimals="4")  # type: ignore[arg-type]
    assert pol.price_decimals == 4
    bid, ask = quote_prices("100.0", "0.01", decimals="2")  # type: ignore[arg-type]
    assert (bid, ask) == (99.0, 101.0)
    exit_px = anchored_exit_price("S", "100.0", "100.5", "0.01", "0.0115", decimals=2)  # type: ignore[arg-type]
    assert exit_px >= 101.16

    # 4. PositionExitConfig normalizes string booleans/floats and PositionExitManager strips symbol keys
    cfg = PositionExitConfig(
        tp_trend_ride_enabled="false",  # type: ignore[arg-type]
        allow_external_cache="0",  # type: ignore[arg-type]
        gain_threshold="0.08",  # type: ignore[arg-type]
    )
    assert cfg.tp_trend_ride_enabled is False
    assert cfg.allow_external_cache is False
    assert cfg.gain_threshold == pytest.approx(0.08)

    pem = PositionExitManager(config=cfg)
    snap = PositionSnapshot(symbol=" BTCUSDT ", current_price="100.0", avg_buy_price="90.0", trend_up=True)  # type: ignore[arg-type]
    assert snap.symbol == "BTCUSDT"
    snap.current_price = "108.0"  # type: ignore[assignment]
    dec = pem.evaluate_take_profit(snap)
    assert dec.should_exit is True

    # 5. verify_tools/portfolio_snapshot._is_paper handles NaN float safely
    assert ps._is_paper(float("nan")) is False
    assert ps._is_paper(1.0) is True


def test_batch26_orchestrator_protobuf_monitor_reconciler_and_gateway_hardening(tmp_path, monkeypatch):
    import asyncio
    from collections import defaultdict, deque
    from types import SimpleNamespace
    import orchestratorOS.agy.protobuf as pb
    from orchestratorOS.livecheck.autonomous_overnight_monitor import AutonomousOvernightMonitor
    from orchestratorOS.supervisor.autonomous_ai_reconciler import AutonomousAIReconciler
    from orchestratorTrade.autonomous_intent_gateway import AutonomousIntentGateway
    from orchestratorTrade.orchestrator import BotManager

    # 1. protobuf helpers handle huge negative varints, non-bytes inputs, and non-str titles without crashing
    assert len(pb.enc_varint(-1)) == 10
    assert len(pb.enc_varint(-(1 << 65) - 1)) == 10
    assert pb.enc_varint(-(1 << 65)) == b"\x00"
    assert pb.summary_ui_visible(None, title=None) == b""  # type: ignore[arg-type]
    assert pb.summary_annotation_title(None) == ""  # type: ignore[arg-type]
    assert pb.entry_cid(None) == ""  # type: ignore[arg-type]
    assert pb.list_entry_cids(None) == set()  # type: ignore[arg-type]
    assert pb.drop_entries(123, {"abc"}) == (b"", 0)  # type: ignore[arg-type]
    assert pb.replace_summary_entry(123, "abc", b"") == (b"", False)  # type: ignore[arg-type]

    # 2. AutonomousOvernightMonitor sanitizes invalid dev_port and string/NaN persist flag
    mon = AutonomousOvernightMonitor(workspace_dir=str(tmp_path), dev_port=True)  # type: ignore[arg-type]
    assert mon.dev_port == 32238
    mon2 = AutonomousOvernightMonitor(workspace_dir=str(tmp_path), dev_port=99999)
    assert mon2.dev_port == 32238

    # 3. AutonomousAIReconciler and AutonomousIntentGateway reject NaN notify_enabled and boolean symbols/venues
    (tmp_path / "order_guard.conf").write_text(
        "supervisor_mode=shadow\n"
        "supervisor_notify=0\n"
        "autonomous_trading_mode=shadow\n"
        "autonomous_trading_min_notional_usd=10\n"
        "autonomous_trading_max_notional_usd=100\n"
        "autonomous_trading_min_confidence=0.8\n"
        "autonomous_trading_notify=0\n",
        encoding="utf-8",
    )
    (tmp_path / "config.env").write_text("NTFY_TOPIC_GUARD=test-guard\n", encoding="utf-8")

    with pytest.raises(ValueError, match="notify_enabled"):
        AutonomousAIReconciler(workspace_dir=str(tmp_path), notify_enabled=float("nan"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="notify_enabled"):
        AutonomousIntentGateway(workspace_dir=str(tmp_path), notify_enabled=float("nan"))  # type: ignore[arg-type]

    gw = AutonomousIntentGateway(workspace_dir=str(tmp_path))
    assert gw.submit({"decision": "BUY", "symbol": True, "venue": "binance", "confidence": 0.95}, source="test", max_mode="shadow") is None
    assert gw.submit({"decision": "BUY", "symbol": "BTCUSDC", "venue": False, "confidence": 0.95}, source="test", max_mode="shadow") is None

    rec = AutonomousAIReconciler(workspace_dir=str(tmp_path), intent_gateway=gw)
    assert rec._execute_venue_or_fleet_action(None) is False  # type: ignore[arg-type]
    assert rec._execute_venue_or_fleet_action({"action": "RESTART_BOT", "symbol": True}) is False

    # 4. BotManager._read_stream continues draining stream even if log_file.write or server.process_line raises
    mgr = object.__new__(BotManager)
    mgr.verbose_stdout = False
    mgr.recent_lines = defaultdict(lambda: deque(maxlen=20))

    class _BadLog:
        def write(self, _s):
            raise OSError("disk full")

        def flush(self):
            raise OSError("disk full")

    class _BadServer:
        def process_line(self, _line, _bot):
            raise RuntimeError("boom")

    mgr.server = _BadServer()

    async def _run_reader():
        reader = asyncio.StreamReader()
        reader.feed_data(b"hello world\nsecond line\n")
        reader.feed_eof()
        await mgr._read_stream(reader, "test_bot", _BadLog())

    asyncio.run(_run_reader())
    assert list(mgr.recent_lines["test_bot"]) == ["hello world", "second line"]


def test_batch27_research_syspath_and_ml_forecast_hardening():
    import numpy as np
    import sys
    import offline.research.hybrid_reentry.harness as hr_harness
    import kraken.backtest_adaptive as kba
    from offline.research.ml_forecast.forecast import build_dataset
    from offline.research.ml_forecast.vol_chronos import realized_vol_series

    # 1. Importing research/kraken modules does not place kraken/ at sys.path[0]
    assert not sys.path[0].endswith("/kraken")

    # 2. kraken.backtest_adaptive.trailing_vol_series handles invalid closes/sample_rate_sec
    assert len(kba.trailing_vol_series(None)) == 0  # type: ignore[arg-type]
    series = kba.trailing_vol_series([100.0 + i * 0.5 for i in range(30)], sample_rate_sec=False)  # type: ignore[arg-type]
    assert len(series) == 30 and np.isfinite(series[-1])

    # 3. ml_forecast.vol_chronos.realized_vol_series and forecast.build_dataset ignore inf/negative prices
    rv = realized_vol_series(np.array([100.0] * 25 + [float("inf"), 102.0]), win=5)
    assert np.isnan(rv[-1])
    X, y_dir, y_mag = build_dataset(np.array([[100.0 + i * 0.1 for i in range(270)]]))
    assert X.shape[1] == 11 and len(X) == len(y_dir) == len(y_mag)


def test_batch28_providers_spot_engine_and_visuals_hardening(monkeypatch):
    from types import SimpleNamespace
    from providers.hyperliquid_provider import HyperliquidProvider
    from providers.kraken_provider import KrakenProvider
    from providers.t212_provider import T212Provider
    from providers.strategy_executor import ProviderError
    from strategies.spot_engine import StratParams
    import visuals.drawprices as dp
    import visuals.tradeall_observe as tobs

    # 1. HyperliquidProvider.order_status normalizes uppercase/whitespace venue_status ("FILLED" / "OPEN")
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x1234567890abcdef")
    hl = HyperliquidProvider(token="HYPE")
    hl._spot_pair = "@107"
    hl._client_tried = True
    hl._client = SimpleNamespace(
        info=SimpleNamespace(
            query_order_by_oid=lambda _addr, _oid: {
                "status": "order",
                "order": {
                    "status": " FILLED ",
                    "order": {"origSz": "2.0", "sz": "0.0"},
                },
            },
            user_fills=lambda _addr: [
                {"oid": 777, "sz": "2.0", "px": "25.0", "fee": "0.05", "feeToken": "USDC"}
            ],
        )
    )
    st = hl.order_status("HYPEUSDC", "777")
    assert st.status == "closed"
    assert st.filled_qty == pytest.approx(2.0)

    # 2. KrakenProvider.get_price_history rejects empty/bool symbol and sorts descending rows
    kp = KrakenProvider(
        client=SimpleNamespace(
            _public=lambda _ep, _params: {
                "XXBTZUSD": [
                    [1999999900, "1", "1", "1", "65000.0"],
                    [1999999800, "1", "1", "1", "64000.0"],
                ]
            }
        )
    )
    assert kp.get_price_history("", 24.0) is None
    assert kp.get_price_history(True, 24.0) is None  # type: ignore[arg-type]
    hist = kp.get_price_history(" BTCUSD ", 24.0)
    assert hist is not None and [h["price"] for h in hist] == [64000.0, 65000.0]

    # 3. T212Provider._send_order rejects invalid symbol and order_status handles bool symbol
    t212 = T212Provider(
        client=SimpleNamespace(
            get_order_status=lambda _oid: {
                "ticker": "NVDA_US_EQ",
                "status": "FILLED",
                "filledQuantity": 1.0,
                "filledValue": 120.0,
                "fee": 0.0,
            }
        )
    )
    with pytest.raises(ProviderError, match="invalid T212 symbol"):
        t212._send_order("   ", "BUY", 1.0, 100.0, market=True)
    t212_st = t212.order_status(True, "901")  # type: ignore[arg-type]
    assert t212_st.status == "closed"

    # 4. StratParams.__post_init__ normalizes string enable_takeprofit and string tp_tranches
    sp = StratParams(
        currency="USD",
        entry_amount=100.0,
        entry_discount_pct=1.0,
        dca_amount=100.0,
        dca_drop_pct=2.0,
        check_minutes=1.0,
        takeprofit_pct=0.0,
        max_budget=500.0,
        max_dca_buys=3,
        enable_takeprofit="false",  # type: ignore[arg-type]
        order_ttl_min=10.0,
        tp_tranches="3:50,6:50",  # type: ignore[arg-type]
    )
    assert sp.enable_takeprofit is False
    assert sp.tp_tranches == [(3.0, 50.0), (6.0, 50.0)]

    # 5. visuals/drawprices and visuals/tradeall_observe edge-case hardening
    assert len(dp.min_max_inverse(None, 0.0, 1.0)) == 0  # type: ignore[arg-type]
    X_seq, y_seq = dp.make_sequences(None, 5)  # type: ignore[arg-type]
    assert X_seq.shape == (0, 5, 1) and y_seq.shape == (0,)
    assert tobs.build_trend_regions(None, 100.0, 200.0) == []
    assert tobs.build_trend_regions([{"ts": 120.0, "state": "UP"}, {"bad": 1}], 100.0, 200.0) == [(120.0, 200.0, "UP")]


def test_batch29_cross_venue_module_collision_and_syspath_isolation():
    import importlib.util
    import os
    import sys
    from shadow_runtime import prepare_shadow_runtime

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hl_dir = os.path.join(root_dir, "hyperliquid")
    t212_dir = os.path.join(root_dir, "212trading")
    gate_common_path = os.path.join(root_dir, "offline", "research", "tradeall_trigger_gate", "common.py")

    # 1. prepare_shadow_runtime keeps root ahead of module_dirs in sys.path
    saved_path = list(sys.path)
    try:
        prepare_shadow_runtime(root_dir, hl_dir, t212_dir)
        assert sys.path.index(root_dir) < sys.path.index(hl_dir)
        assert sys.path.index(root_dir) < sys.path.index(t212_dir)
    finally:
        sys.path[:] = saved_path

    # 2. Even if offline/research/tradeall_trigger_gate/common.py is cached as sys.modules["common"],
    # hyperliquid modules resolve hyperliquid/common.py and do not raise ImportError.
    prev_common = sys.modules.get("common")
    try:
        gate_spec = importlib.util.spec_from_file_location(
            "offline_trigger_gate_common_probe", gate_common_path
        )
        gate_mod = importlib.util.module_from_spec(gate_spec)
        # Stub minimal module with __file__ pointing to offline/research/tradeall_trigger_gate/common.py
        gate_mod.__file__ = gate_common_path
        sys.modules["common"] = gate_mod

        for rel_file in ("hl_client.py", "market_data.py", "price_analysis.py", "signals.py", "hl_bot.py", "backtest.py"):
            sys.modules["common"] = gate_mod
            spec = importlib.util.spec_from_file_location(
                f"_test_hl_isolated_{rel_file[:-3]}", os.path.join(hl_dir, rel_file)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            resolved_common = sys.modules.get("common")
            assert resolved_common is not None
            assert os.path.dirname(os.path.abspath(resolved_common.__file__)) == hl_dir
    finally:
        if prev_common is not None:
            sys.modules["common"] = prev_common
        else:
            sys.modules.pop("common", None)


def test_batch30_trigger_gate_common_grid_reports_and_monitortrades_pilot_hardening(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from offline.research.tradeall_trigger_gate.common import (
        make_instrumented_trend_state,
        make_tracker_logic,
    )
    from offline.research.kraken_continuous_grid import report as k_rep1
    from offline.research.kraken_continuous_grid import report3 as k_rep3
    from offline.research.monitortrades_backtest import scheduled_pilot as sp

    # 1. make_instrumented_trend_state handles None stats and boolean confirm_threshold_override safely
    ts_cls = make_instrumented_trend_state(
        None,  # type: ignore[arg-type]
        confirm_threshold_override=True,  # type: ignore[arg-type]
        retry_interval_sec=float("nan"),
    )
    inst_ts = ts_cls(60.0, 160.0, 30.0)
    inst_ts.start_trend("UP")
    assert inst_ts.can_retry("UP", 1000.0) is True
    inst_ts.mark_attempt("UP", 1000.0)
    assert inst_ts.can_retry("UP", 1100.0) is False
    assert inst_ts.can_retry("UP", 2900.0) is True

    # 2. make_tracker_logic handles invalid retry_interval_sec, non-dict tr.stats, and non-finite _now()
    fake_tr = SimpleNamespace(
        stats=None,
        fired_up=False,
        fired_down=False,
        last_attempt_up_ts=100.0,
        last_attempt_down_ts=None,
        update=lambda _ts, _px: 1,
    )
    logic_fn = make_tracker_logic("TEST", lambda _sym: fake_tr, retry_interval_sec=-5.0)
    logic_fn(
        None, False, "BTCUSDC", 0.0, 0.0,
        SimpleNamespace(_now=lambda: "bad_ts"),
        60000.0,
    )
    assert fake_tr.last_attempt_up_ts == 100.0

    # 3. kraken_continuous_grid report.py and report3.py handle malformed/null nested full/year/0.26 payloads
    exp_dir = tmp_path / "kgrid"
    exp_dir.mkdir()
    (exp_dir / "results.jsonl").write_text(
        json.dumps({
            "cfg": "live",
            "asset": "BTC",
            "full": {"ret": None, "dd": "5.0", "expo": None, "bh": None},
            "year": {"2024": {"ret": None, "bh": None}, "2025": "bad_year_entry"},
            "w90": [[1.5, 2.0, 3.0]],
            "w30": [[0.5, 1.0, 1.0]],
        }) + "\n" + json.dumps({
            "cfg": "ov650_t8",
            "asset": "BTC",
            "full": "not_a_dict",
            "year": None,
        }) + "\n",
        encoding="utf-8",
    )
    lines1 = k_rep1.generate_report(str(exp_dir))
    assert any("| BTC |" in ln for ln in lines1)

    (exp_dir / "results3.jsonl").write_text(
        json.dumps({
            "cfg": "live",
            "asset": "BTC",
            "0.26": {"ret": None, "dd": None, "refused": None},
            "0.4": {"ret": 5.0, "dd": 2.0, "refused": 0},
        }) + "\n" + json.dumps({
            "cfg": "sl0",
            "asset": "BTC",
            "0.26": "not_a_dict",
        }) + "\n",
        encoding="utf-8",
    )
    lines3 = k_rep3.generate_report(str(exp_dir))
    assert any("| BTC |" in ln for ln in lines3)

    # 4. scheduled_pilot._current_value raises clear KeyError on missing section/key
    conf_file = tmp_path / "instruments.conf"
    conf_file.write_text("[BINANCE_BTC]\nmt.gain = 7.0\n", encoding="utf-8")
    monkeypatch.setattr(sp, "INSTRUMENTS_CONF", str(conf_file))
    assert sp._current_value("BINANCE_BTC", "mt.gain") == pytest.approx(7.0)
    with pytest.raises(KeyError, match="not found"):
        sp._current_value("BINANCE_BTC", "mt.missing_key")



