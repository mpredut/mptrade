"""Targeted verification suite for Wave 4 backtesting & research bug fixes."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import sklearn.ensemble as sk_ens

from kraken import backtest as kr_bt, shadow_live as kr_sl
from offline.backtests.datasets import load_dataset, normalize_record
from offline.backtests.evaluation import evaluate_walk_forward, load_venue_module
from offline.backtests.execution import validate_ohlc_bars
from offline.research import comprehensive_strategy_sweep as css
from offline.research.comprehensive_surge_sweep import run_deep_surge_sweep as dss
from offline.research.hybrid_reentry import harness, report, run_equities
from offline.research.kraken_continuous_grid import report as k_rep, report2 as k_rep2
from offline.research.ml_forecast import forecast as fc, vol_chronos as vc
from offline.research.ml_forecast.priceprediction import PricePrediction
from offline.research.monitortrades_backtest import (
    backtest_wait_and_regime as bwr,
    compare_sl_variants as csl,
    replay_trend_source as rts,
    run_replay_backtest as rrb,
    sweep_mt_models as smm,
)
from offline.research.t212_continuous_grid import report as t_rep
import pricewindow as pw
from providers.replay_provider import ReplayMarketDataProvider
from tests.conftest import make_fake_chronos_pipeline


def test_datasets_str_path_validation_and_finite_normalize(tmp_path):
    with pytest.raises(ValueError):
        normalize_record({"timestamp": 1700000000, "open": float("nan"), "high": 10.0, "low": 9.0, "close": 9.5})

    good_csv = tmp_path / "good.csv"
    good_csv.write_text(
        "timestamp,open,high,low,close\n"
        "1700000000,100.0,105.0,99.0,102.0\n"
        "1700003600,102.0,106.0,101.0,104.0\n",
        encoding="utf-8",
    )
    rows = load_dataset(str(good_csv))
    assert len(rows) == 2
    assert rows[0]["timestamp"] == 1700000000

    dup_csv = tmp_path / "dup.csv"
    dup_csv.write_text(
        "timestamp,open,high,low,close\n"
        "1700000000,100.0,105.0,99.0,102.0\n"
        "1700000000,102.0,106.0,101.0,104.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(str(dup_csv))


def test_walk_forward_overlapping_test_folds_null_compounded_return():
    records = [
        {"timestamp": 1700000000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
        for i in range(30)
    ]
    dummy_replay = lambda ohlc, warmup_ohlc, context: {
        "return_pct": 10.0,
        "max_drawdown_pct": 2.0,
        "cycles": 1,
        "fills": 2,
    }
    over_rep = evaluate_walk_forward(
        records, dummy_replay, train_size=10, validation_size=5, test_size=8, step_size=4, warmup_bars=2
    )
    assert over_rep["aggregate_test"]["compounded_reset_return_pct"] is None

    disj_rep = evaluate_walk_forward(
        records, dummy_replay, train_size=10, validation_size=4, test_size=8, step_size=8, warmup_bars=2
    )
    assert disj_rep["aggregate_test"]["compounded_reset_return_pct"] == pytest.approx(21.0)


def test_execution_validate_ohlc_bars_error_distinction():
    with pytest.raises(ValueError, match="non-numeric prices"):
        validate_ohlc_bars([("bad", 10.0, 9.0, 9.5)])
    with pytest.raises(ValueError, match="has non-finite or invalid prices"):
        validate_ohlc_bars([(float("nan"), 10.0, 9.0, 9.5)])
    with pytest.raises(ValueError, match="has non-finite or invalid prices"):
        validate_ohlc_bars([(10.0, 9.0, 9.5, 10.0)])


def test_t212_replay_initial_cash_and_backtest_param_unmasking(monkeypatch):

    t212_replay = load_venue_module("212trading", "replay.py", "test_w4_t212_replay")
    t212_bt = load_venue_module("212trading", "backtest.py", "test_w4_t212_bt")
    StratParams = t212_replay._strat.StratParams

    env_cfg = dict(t212_bt._SIMULATE_ENV_DEFAULTS)
    env_cfg.update(
        {
            "STRATEGY_MODE": "avg_tp",
            "STRAT_CURRENCY": "USD",
            "YAHOO_SYMBOL": "TEST",
            "STRAT_ENTRY_PCT": "50",
            "STRAT_DCA_PCT": "50",
            "STRAT_ENTRY_DISCOUNT_PCT": "0.0",
            "STRAT_DCA_DROP_PCT": "2.0",
            "STRAT_TAKEPROFIT_PCT": "5.0",
            "STRAT_MAX_DCA_BUYS": "3",
            "STRAT_MAX_BUDGET": "1000",
            "STRAT_FX_FEE_PCT": "0.15",
            "STRAT_STOP_LOSS_PCT": "0.0",
            "STRAT_CHECK_MINUTES": "60",
            "STRAT_ORDER_TTL_MIN": "120",
        }
    )
    params = StratParams.from_env(env_cfg)
    ohlc = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 100.5, 96.0, 97.0),
        (97.0, 98.0, 93.0, 94.0),
        (94.0, 105.0, 94.0, 104.0),
    ]
    constrained = t212_replay.run_replay(ohlc, params, bar_minutes=60.0, initial_cash=400.0)
    assert constrained["initial_capital"] == 400.0
    assert constrained["cash_refusals"] >= 1
    assert constrained["min_cash"] >= -1e-6

    unconstrained = t212_replay.run_replay(ohlc, params, bar_minutes=60.0)
    assert unconstrained["initial_capital"] == 1000.0
    assert "cash_refusals" not in unconstrained

    # Verify main() preserves "auto" max_dca_buys and unmasks tp_ladder/tp_mdhtp_enabled when --tp is passed
    seen_sim = []
    monkeypatch.setattr(t212_bt, "fetch_candles", lambda sym, rng, itv: [(100.0, 101.0, 99.0, 100.0)] * 25)
    monkeypatch.setattr(
        t212_bt,
        "simulate",
        lambda ohlc_arg, p_dict: (seen_sim.append(dict(p_dict)) or {"total": 0.0, "net": 0.0, "max_drawdown_pct": 0.0, "cycles": 0, "wins": 0, "open_qty": 0.0, "fills": 0}),
    )
    monkeypatch.setattr("sys.argv", ["backtest.py", "--profile", "rgnt", "--tp", "15.0"])
    rc = t212_bt.main()
    assert rc == 0
    assert seen_sim[-1]["tp"] == 15.0
    assert seen_sim[-1]["tp_ladder"] == ""
    assert seen_sim[-1]["tp_mdhtp_enabled"] is False
    assert seen_sim[-1]["maxdca"] >= 1


def test_kraken_backtest_surge_dynamic_and_shadow_live_warmup_and_hl_restore(monkeypatch):
    p60 = kr_bt._bt_params({"interval": 60})
    if p60.tp_trail_vol_interval != 60:
        assert p60.surge_dynamic is False

    seen_kwargs = []
    monkeypatch.setattr(
        kr_sl,
        "_variants",
        lambda interval: {"current": SimpleNamespace(allocated_budget=lambda: 1000.0)},
    )

    def _spy_run_one(ohlc, params, interval, fee_pct, **kw):
        seen_kwargs.append(dict(kw))
        return {"net": 10.0, "total": 10.0, "max_drawdown_pct": 1.0, "cycles": 1, "open_qty": 0.0}

    monkeypatch.setattr(kr_sl, "_run_one", _spy_run_one)
    ohlc = [(100.0, 101.0, 99.0, 100.0), (100.0, 102.0, 99.5, 101.0)]
    kr_sl._eval_block(ohlc, 240, 0.26)
    assert "warmup_ohlc" not in seen_kwargs[-1]
    kr_sl._eval_block(ohlc, 240, 0.26, warmup_ohlc=[(99.0, 100.0, 98.0, 99.5)])
    assert seen_kwargs[-1]["warmup_ohlc"] == [(99.0, 100.0, 98.0, 99.5)]

    hl_lt = load_venue_module("hyperliquid", "shadow_longterm.py", "test_w4_hl_shadow_lt")
    orig_dir = hl_lt.shadow.LOG_DIR
    orig_vars = hl_lt.shadow._variants
    orig_fetch = hl_lt.shadow._fetch_with_ts
    monkeypatch.setattr(hl_lt, "_fetch_with_ts", lambda pair, interval: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        hl_lt.snapshot(quiet=True)
    assert hl_lt.shadow.LOG_DIR == orig_dir
    assert hl_lt.shadow._variants is orig_vars
    assert hl_lt.shadow._fetch_with_ts is orig_fetch


def test_comprehensive_strategy_and_surge_sweeps(tmp_path, monkeypatch):
    grid = css.define_candidate_grid()
    assert "00_LIVE_CURRENT" in grid
    assert "25_COMBO_smart_flow" in grid
    assert "27_COMBO_high_calmar_runner" in grid
    assert "29_COMBO_ultimate_balanced" in grid
    assert grid["01_tp4"]["tp_dynamic_flat"] is False
    assert grid["01_tp4"]["tp_mdhtp_enabled"] is False
    materialized = [dataclasses.replace(css.LIVE_PARAMS, **ov) for ov in grid.values()]
    assert len({repr(p) for p in materialized}) == len(grid)

    data_dir = tmp_path / "surge_data"
    out_dir = tmp_path / "surge_out"
    data_dir.mkdir()
    monkeypatch.setattr(dss, "ASSETS", ["BTC"])
    csv_lines = ["timestamp,open,high,low,close"] + [
        f"{1700000000 + i * 14400},100.0,101.0,99.0,100.5" for i in range(80)
    ]
    (data_dir / "BTC_240m.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        dss,
        "build_candidate_grid",
        lambda base_p: [
            {"id": "BASELINE_SURGE_OFF", "category": "Baseline", "desc": "off", "params": dataclasses.replace(base_p, surge_guard=False)},
            {"id": "DYN_WIN_CAND", "category": "Dynamic", "desc": "dyn", "params": dataclasses.replace(base_p, surge_guard=True, surge_dynamic=True, surge_move_pct=15.0, surge_exit_pullback_pct=2.5, reentry_bear_bounce_pct=1.5)},
        ],
    )

    def _fake_eval_cand(cand, asset_data, initial_cash, fee_pct):
        is_win = cand["id"] == "DYN_WIN_CAND"
        return {
            "id": cand["id"],
            "category": cand["category"],
            "desc": cand["desc"],
            "total_profit_usd": 150.0 if is_win else 100.0,
            "h1_profit_usd": 75.0 if is_win else 50.0,
            "h2_profit_usd": 75.0 if is_win else 50.0,
            "worst_maxdd_pct": 5.0,
            "pnl_per_drawdown_point": 30.0 if is_win else 20.0,
            "total_cycles": 4,
            "asset_profits": {"BTC": 150.0 if is_win else 100.0},
            "asset_maxdds": {"BTC": 5.0},
            "strategy_params": dataclasses.asdict(cand["params"]),
        }

    monkeypatch.setattr(dss, "evaluate_single_candidate", _fake_eval_cand)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_deep_surge_sweep.py",
            "--workers",
            "1",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--fee-pct",
            "0.26",
        ],
    )
    dss.main()
    saved = json.loads((out_dir / "sweep_results.json").read_text(encoding="utf-8"))
    assert saved["failed_candidates"] == 0
    assert saved["baseline"]["beats_baseline_both_halves"] is False
    md = (out_dir / "DEEP_SURGE_SWEEP_REPORT.md").read_text(encoding="utf-8")
    assert "STRAT_SURGE_GUARD=true" in md
    assert "DYN_WIN_CAND" in md


def test_hybrid_reentry_rgnt_profile_preservation_and_timestamps(tmp_path, monkeypatch):
    csv_file = tmp_path / "RGNT_1d.csv"
    lines = ["timestamp,open,high,low,close"]
    for i in range(40):
        lines.append(f"{1700000000 + i * 86400},100.0,102.0,98.0,101.0")
    csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ohlc, ts = harness.load_ohlc_with_timestamps_csv(csv_file)
    assert len(ohlc) == 40 and len(ts) == 40 and ts[0] == 1700000000

    seen_calls = []
    monkeypatch.setattr(run_equities, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(run_equities, "DATASETS", {"RGNT": str(csv_file)})

    def _fake_eval(*, asset_name, ohlc, base_params, **kw):
        seen_calls.append((asset_name, base_params, kw))
        return [{"asset": asset_name, "test_id": "Test 0", "variant": "Baseline", "total_return_pct": 5.0, "max_drawdown_pct": 2.0, "exposure_pct": 30.0, "cycles": 1}]

    monkeypatch.setattr(run_equities, "evaluate_asset", _fake_eval)
    run_equities.main()
    assert len(seen_calls) == 1
    _, rgnt_params, call_kw = seen_calls[0]
    # RGNT profile has TP=35%, DCA_DROP=40%, SL=50% (must NOT be clobbered by BASE_CONFIG 8%/1.3%/25%)
    assert rgnt_params.takeprofit_pct == pytest.approx(35.0)
    assert rgnt_params.dca_drop_pct == pytest.approx(40.0)
    assert rgnt_params.stop_loss_pct == pytest.approx(50.0)
    assert call_kw["initial_cash"] == 500.0
    assert len(call_kw["timestamps"]) == 40

    # Report filters error rows cleanly
    crypto_json = tmp_path / "ablation_results.json"
    summary_json = tmp_path / "ablation_summary.json"
    eq_json = tmp_path / "equities_ablation_results.json"
    crypto_json.write_text(json.dumps([{"asset": "BTC", "test_id": "Test 0", "variant": "Baseline", "total_return_pct": 10.0, "max_drawdown_pct": 5.0}]), encoding="utf-8")
    summary_json.write_text(json.dumps({"Test 0": {"name": "Baseline"}}), encoding="utf-8")
    eq_json.write_text(
        json.dumps(
            [
                {"asset": "RGNT", "test_id": "Test 0", "variant": "Baseline", "total_return_pct": 5.0, "max_drawdown_pct": 2.0, "exposure_pct": 30.0, "cycles": 1},
                {"asset": "BAD_EQ", "test_id": "Test 0", "error": "failed"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "CRYPTO_JSON", str(crypto_json))
    monkeypatch.setattr(report, "SUMMARY_JSON", str(summary_json))
    monkeypatch.setattr(report, "EQUITIES_JSON", str(eq_json))
    rep_txt = report.generate_markdown_report()
    assert "RGNT" in rep_txt
    assert "BAD_EQ" not in rep_txt


def test_ml_forecast_purge_chronos_tail_and_priceprediction_isolation(monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, len(fc.FEATURES)))
    y_mag = rng.normal(scale=0.02, size=80)
    fit_lengths = []

    orig_fit = sk_ens.HistGradientBoostingRegressor.fit

    def _wrapped_fit(self, X_fit, y_fit, *a, **kw):
        fit_lengths.append(len(X_fit))
        return orig_fit(self, X_fit, y_fit, *a, **kw)

    monkeypatch.setattr(sk_ens.HistGradientBoostingRegressor, "fit", _wrapped_fit)
    y_dir = (y_mag > 0).astype(int)
    rep = fc.walk_forward(X, y_dir, y_mag, train_frac=0.7, refit_every=50)
    assert rep["n_test"] == 80 - 56
    assert fit_lengths[0] == 56 - fc.HORIZON_H

    seg = np.array([0.1, 0.2, np.nan, 0.3, 0.4], dtype=float)
    assert list(vc._clean_tail(seg)) == [0.3, 0.4]

    monkeypatch.setattr(vc, "_load_pipeline", lambda: make_fake_chronos_pipeline(1.0, last_col_value=9.0))
    monkeypatch.setattr(vc, "torch", SimpleNamespace(float32="f32", tensor=lambda a, dtype=None: np.asarray(a)))
    rv = np.linspace(0.1, 0.5, 40)
    wf_vol = vc.walk_forward_vol(rv, horizon=5, warmup=5, stride=10)
    assert wf_vol["n_test"] > 0
    assert wf_vol["mae_model"] > 8.0

    pp = PricePrediction(window_size=4, epochs=5)
    pp.process_prices([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1000.0])
    assert float(pp.scaler.data_max_[0]) == pytest.approx(16.0)


def test_monitortrades_backtest_signals_wait_and_order_cache_immutability():
    series = [(1000.0 + i, 100.0 + np.sin(i * 0.2) + i * 0.01) for i in range(300)]
    s_default_tao = smm.compute_all_signals(series, win_sec=222.0, symbol="TAOUSDC", use_symbol_fast_window=True)
    s_explicit_222 = smm.compute_all_signals(series, win_sec=222.0, symbol="TAOUSDC", use_symbol_fast_window=False)
    if pw.fast_trend_window_for_symbol("TAOUSDC") == 90.0:
        assert not np.array_equal(s_default_tao["m"], s_explicit_222["m"])

    ts = np.array([t for t, _ in series], dtype=np.float64)
    px = np.array([p for _, p in series], dtype=np.float64)
    a, sr = smm.compute_window_bounds(ts, win_sec=222.0)
    direct = smm.compute_window_signals_from_bounds(px, a, sr, symbol="BTCUSDC", recent_win_sec=222.0, min_valid_n=2)
    via_all = smm.compute_all_signals(series, win_sec=222.0, symbol="BTCUSDC", use_symbol_fast_window=False)
    assert np.allclose(direct["gr"], via_all["gr"])

    sig_w = bwr.compute_exact_time_window_signals(ts[:10], px[:10], 5.0, "BTCUSDC")
    assert sig_w["legacy"][0] == 0.0 and sig_w["legacy"][1] == 0.0

    sparse_ts = np.array([0.0, 10.0, 500.0, 510.0], dtype=np.float64)
    sparse_px = np.array([100.0, 99.0, 50.0, 50.0], dtype=np.float64)
    wb = np.array([True, True, True, True], dtype=bool)
    wait_out = bwr.simulate_should_wait(sparse_ts, sparse_px, wb, wb, max_wait_sec=30.0, stride=4)
    assert wait_out["avg_wait_s"] <= 30.0

    src = rts.ReplayTrendSource(["BTCUSDC"], window_seconds=222.0)
    for i in range(100):
        src.advance("BTCUSDC", float(i), 100.0 + (0.01 if i % 2 == 0 else -0.008) + i * 0.002)
    assert isinstance(src.should_wait("BUY", "BTCUSDC", use_noise_gate=True), bool)

    prov = ReplayMarketDataProvider({"BTCUSDC": [(float(i), 100.0 + i * 0.5) for i in range(1, 10)]})
    prov.advance("BTCUSDC")
    prov.place_order("BTCUSDC", "BUY", 100.5, 1.0)
    prov.advance("BTCUSDC")
    prov.place_order("BTCUSDC", "BUY", 101.0, 1.0)
    orders_before = prov.get_orders("BTCUSDC", "BUY", 100.0)
    assert [o["timestamp"] for o in orders_before] == [1000, 2000]
    res_sym = rrb.run_symbol(
        "BTCUSDC",
        {"mt.gain": "5.0", "mt.lost": "3.0", "mt.maxage_days": "7"},
        "BTC",
        quiet=True,
        series=[(float(i), 100.0 + (i % 5) * 1.5) for i in range(1, 40)],
    )
    assert res_sym is not None and "net" in res_sym
    assert csl._series_span_sec([(100.0, 1.0), (500.0, 2.0)]) == 400.0


def test_grid_reports_missing_live_and_strict_json(tmp_path):
    k_dir = tmp_path / "kgrid"
    k_dir.mkdir()
    (k_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "cfg": "cand1",
                "asset": "BTC",
                "full": {"ret": 12.0, "dd": 5.0, "expo": 40.0, "bh": 8.0},
                "w90": [],
                "w30": [],
                "year": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lines = k_rep.generate_report(str(k_dir))
    md = "\n".join(lines)
    assert "nan" not in md.lower()
    assert "| cand1 | 0/1 | 0/1 |" in md
    summary_raw = (k_dir / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_raw
    parsed = json.loads(summary_raw)
    assert parsed[0]["beat_full"] == 0
    assert parsed[0]["w90_mean"] is None

    (k_dir / "results2.jsonl").write_text(
        json.dumps(
            {
                "kind": "views",
                "cfg": "cand1",
                "fee": "std",
                "asset": "BTC",
                "full": {"ret": 12.0, "dd": 5.0},
                "w90": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lines2 = k_rep2.generate_report(str(k_dir))
    md2 = "\n".join(lines2)
    assert "nan" not in md2.lower()
    assert "| cand1 | std | 0/1 |" in md2

    t_dir = tmp_path / "tgrid"
    t_dir.mkdir()
    (t_dir / "results.jsonl").write_text(
        json.dumps({"profile": "nvda", "variant": "live", "asset": "NVDA", "1h": {"ret": 10.0, "dd": 4.0, "expo": 50.0, "cyc": 3}})
        + "\n",
        encoding="utf-8",
    )
    t_md = t_rep.generate_report(str(t_dir))
    assert "None" not in t_md
    assert "| live | 10.0 | 4.0 | 50.0 | 3 | - | - | - | - |" in t_md
