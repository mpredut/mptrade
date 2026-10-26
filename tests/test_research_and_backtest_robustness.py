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
    assert lt.window_sec == 60.0
    assert lt.reeval_sec == 1.0
    assert lt.min_points == 2
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




