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
    assert pw.get_instant_trend()[3] == 0.0
    assert abs(pw.get_instant_trend()[0] - instant_trend_from_prices(pw.prices, sample_rate_sec=15.0)[0]) < 1e-9


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

    c24_file = tmp_path / "cache_24price_ARBUSDC.json"
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

    # Verify execute_tradeall_variant restores all patched tradeall attributes even after run
    c24_file = tmp_path / "cachedb" / "cache_365price_ARBUSDC.jsonl"
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

    cache_file = tmp_path / "cache_price_sparse_TESTUSDC.jsonl"
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
