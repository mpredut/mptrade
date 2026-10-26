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

    # 4. T212Provider._signed_qty rejects non-positive quantities
    try:
        T212Provider._signed_qty("BUY", -1.0)
        raise AssertionError("Expected ProviderError for negative T212 quantity")
    except ProviderError:
        pass


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
    noisy_prices = [None, True] + [100.0 + (i % 3) * 0.5 for i in range(25)] + ["bad", False]
    v1h = calculate_volatility_1h(noisy_prices, 60.0)
    assert v1h is not None and v1h > 0.0
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


