"""Targeted regression tests for Wave 8 backtest, replay, strategy, research, and runner fixes."""

import ast
import dataclasses
from datetime import datetime, timedelta, timezone
import subprocess
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from offline.backtests.evaluation import load_venue_replay_module
from offline.research import comprehensive_strategy_sweep as css, slice_dense_archive as sda
from offline.research.hybrid_reentry import harness as hr_harness
from offline.research.ml_forecast import forecast as ml_fc, vol_chronos as ml_vc
from offline.research.ml_forecast.priceprediction import PricePrediction
from offline.research.monitortrades_backtest import (
    backtest_wait_and_regime as bwr,
    replay_trend_source as rts,
    run_replay_backtest as rrb,
    scheduled_pilot as sp,
)
from offline.research.t212_continuous_grid import grid as t212_grid, report as t212_report
from offline.runners import financial_promotion_gate, walk_forward_compare
from providers.replay_provider import load_price_series
from state_io import atomic_write_json, atomic_write_jsonl, atomic_write_text, iter_jsonl_dicts, read_text_file
from strategies import spot_engine, spot_replay, spot_rules
from strategies.base_spot_mixin import BaseSpotMixin, sanitize_common_spot_state
from strategies.state_store import (
    compute_adaptive_reentry_pullback_pct,
    safe_state_token,
    update_post_sell_anchors,
)
from tests.conftest import load_venue_modules, make_fake_chronos_pipeline

ROOT = Path(__file__).resolve().parents[1]
_KRAKEN_BT, _KRAKEN_COMMON = load_venue_modules("kraken", "backtest", "kraken_common")
_T212_BT = load_venue_modules("212trading", "backtest")
_T212_REPLAY = load_venue_replay_module("t212", "test_w8_t212_replay")

WAVE8_1_PY_FILES = (
    "strategies/base_spot_mixin.py",
    "strategies/spot_engine.py",
    "strategies/spot_rules.py",
    "strategies/state_store.py",
    "strategies/spot_replay.py",
    "212trading/replay.py",
    "kraken/backtest.py",
    "212trading/backtest.py",
    "kraken/shadow_live.py",
    "hyperliquid/shadow_longterm.py",
    "offline/runners/financial_promotion_gate.py",
    "offline/runners/walk_forward_compare.py",
    "offline/research/slice_dense_archive.py",
    "offline/research/t212_continuous_grid/grid.py",
    "offline/research/t212_continuous_grid/report.py",
    "offline/research/hybrid_reentry/harness.py",
    "offline/research/hybrid_reentry/run_equities.py",
    "offline/research/comprehensive_strategy_sweep.py",
    "offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py",
    "offline/research/monitortrades_backtest/sweep_multiwindow_mt.py",
    "offline/research/monitortrades_backtest/backtest_wait_and_regime.py",
    "offline/research/monitortrades_backtest/run_replay_backtest.py",
    "offline/research/monitortrades_backtest/replay_trend_source.py",
    "offline/research/monitortrades_backtest/scheduled_pilot.py",
    "offline/research/ml_forecast/forecast.py",
    "offline/research/ml_forecast/priceprediction.py",
    "offline/research/ml_forecast/vol_chronos.py",
    "tests/test_wave8_backtest_fixes.py",
)

WAVE8_1_SH_FILES = (
    "orchestratorOS/lib/backup_sync_common.sh",
    "offline/runners/load_dev_backtest_env.sh",
    "offline/runners/run_dev_regression.sh",
    "offline/runners/run_overnight_master_suite.sh",
    "offline/runners/sync_cache_to.sh",
)


def test_wave8_1_ascii_and_ast_hygiene() -> None:
    """Enforce 100% ASCII and zero raw float/int/open calls across Wave 8.1 Python and shell files."""
    for rel in WAVE8_1_PY_FILES:
        src = read_text_file(str(ROOT / rel))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"float", "int", "open"}, (
                    f"Forbidden raw {node.func.id}() in {rel}:{node.lineno}"
                )
    for rel in WAVE8_1_SH_FILES:
        path = ROOT / rel
        src = read_text_file(str(path))
        assert src.isascii(), f"Non-ASCII content in {rel}"
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"bash -n failed for {rel}: {proc.stderr}"


def test_replay_cache_bounds_and_immutable_regime_cache() -> None:
    """Ensure replay caches are bounded for 4GB VMs and cached closes cannot be mutated by callers."""
    assert BaseSpotMixin._REPLAY_CACHE_MAX == 16384
    assert isinstance(spot_engine.Strategy._REPLAY_REGIME_CACHE, OrderedDict)

    bars = [(100.0 + i * 0.2, 101.0 + i * 0.2, 99.0 + i * 0.2, 100.0 + i * 0.2) for i in range(45)]
    base = spot_engine.StratParams.from_env(
        _KRAKEN_COMMON.load_isolated_kraken_cfg(ROOT / "kraken")
    )
    params = spot_replay.align_spot_interval_params(dataclasses.replace(base, tp_regime_gate=True), 240)
    res1 = spot_replay.run_replay(bars[40:], params, fee_pct=0.26, bar_minutes=240, warmup_ohlc=bars[:40])
    assert spot_engine.Strategy._REPLAY_REGIME_CACHE
    for _dec, cached_closes in spot_engine.Strategy._REPLAY_REGIME_CACHE.values():
        assert isinstance(cached_closes, tuple)
    res2 = spot_replay.run_replay(bars[40:], params, fee_pct=0.26, bar_minutes=240, warmup_ohlc=bars[:40])
    assert res1["total"] == res2["total"]


def test_sanitize_orders_fast_profit_min_ts_and_state_store_fixes() -> None:
    """Verify invalid order sides are dropped, pre-entry shadow peaks are ignored, and state_store clamps hold."""
    state = {
        "orders": [
            {"txid": "ok-1", "side": "BUY", "price": 100.0, "vol": 1.0},
            {"txid": "bad-none", "price": 100.0, "vol": 1.0},
            {"txid": "bad-str", "side": "INVALID", "price": 100.0, "vol": 1.0},
            {"txid": "bad-bool", "side": True, "price": 100.0, "vol": 1.0},
        ]
    }
    cleaned = sanitize_common_spot_state(
        state,
        dict,
        nonneg_float_keys=(),
        signed_float_keys=(),
        nonneg_int_keys=(),
        pos_float_keys=(),
        bool_keys=(),
        dict_keys=(),
        lowercase_order_side=True,
    )
    assert [o["txid"] for o in cleaned["orders"]] == ["ok-1"]
    assert cleaned["orders"][0]["side"] == "buy"

    # Pre-entry peak (t=100, px=130) must NOT trigger fast-profit reversal after entry at t=200 (px=100 -> 112)
    shadows = [(100.0, 130.0), (200.0, 100.0), (250.0, 112.0)]
    trig_without_min_ts, _, _ = spot_rules.check_fast_profit_reversal(
        current_price=112.0, avg_cost=100.0, base_tp_pct=5.0, mult=2.0,
        shadow_prices=shadows, window_sec=600.0, drop_pct=2.0, current_time=260.0,
    )
    trig_with_min_ts, _, drop_after_entry = spot_rules.check_fast_profit_reversal(
        current_price=112.0, avg_cost=100.0, base_tp_pct=5.0, mult=2.0,
        shadow_prices=shadows, window_sec=600.0, drop_pct=2.0, current_time=260.0, min_ts=200.0,
    )
    assert trig_without_min_ts is True
    assert trig_with_min_ts is False
    assert drop_after_entry == 0.0

    # state_store: compute_adaptive_reentry_pullback_pct rejects non-positive/NaN vol and inverted min/max
    p_stub = SimpleNamespace(
        reentry_pullback_adaptive=True,
        reentry_pullback_pct=2.5,
        reentry_pullback_min=4.0,
        reentry_pullback_max=1.0,
        reentry_pullback_k=1.0,
    )
    val_zero, reason_zero = compute_adaptive_reentry_pullback_pct(p_stub, 0.0)
    assert val_zero == 2.5 and "fallback" in reason_zero
    val_clamped, _ = compute_adaptive_reentry_pullback_pct(p_stub, 2.0)
    assert val_clamped == 2.0

    # state_store: update_post_sell_anchors seeds sl_low from last_sell_price when sl_low is None
    st_stop = {"last_sell_price": 90.0, "last_exit_kind": "STOP", "sl_low": None}
    update_post_sell_anchors(st_stop, 95.0, track_stop_sl_low=True)
    assert st_stop["sl_low"] == 90.0

    # state_store: safe_state_token strips leading/trailing '=' and rejects non-ASCII letters
    assert safe_state_token("==NVDA=X==", allow_equals=True) == "NVDA=X"
    assert safe_state_token("===", default="fallback", allow_equals=True) == "fallback"
    assert safe_state_token("BTC\u00e9USD") == "BTCUSD"


def test_t212_replay_final_bar_exit_fx_clamp_and_subbar_timeout_scaling() -> None:
    """Verify T212 replay settles final-bar market exits, clamps FX-drifted limit buys, and scales timeouts."""
    cfg = _T212_BT._load_isolated_t212_cfg(
        str(ROOT / "212trading" / ".env"),
        profile_file=str(ROOT / "212trading" / "config.nvda.env"),
    )
    cfg.update({
        "STRAT_CURRENCY": "USD",
        "STRAT_MAX_BUDGET": "100",
        "STRAT_ENTRY_PCT": "80",
        "STRAT_DCA_PCT": "10",
        "STRAT_ENTRY_DISCOUNT_PCT": "0.1",
        "STRAT_STOP_LOSS_PCT": "10.0",
        "STRAT_SL_MHDSL_ENABLED": "false",
        "STRAT_TP_MDHTP_ENABLED": "false",
        "STRAT_DCA_TREND_GATE_PCT": "0",
    })
    params = _T212_REPLAY._strat.StratParams.from_env(cfg)

    # Final-bar stop-loss triggers a market sell on bar 2 (last bar) and must settle at close
    bars_stop = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.2, 99.0, 100.0),
        (85.0, 86.0, 84.0, 85.0),
    ]
    res_stop = _T212_REPLAY.run_replay(bars_stop, params, bar_minutes=60.0, initial_cash=100.0)
    assert res_stop["open_qty"] == 0.0
    assert res_stop["cycles"] == 1
    assert res_stop["net"] < 0.0

    # Adverse FX move between bar 0 placement (fx=1.0) and bar 1 fill (fx=0.98) clamps qty instead of canceling
    cfg_eur = dict(cfg, STRAT_CURRENCY="EUR", STRAT_FX_FEE_PCT="0.15")
    params_eur = _T212_REPLAY._strat.StratParams.from_env(cfg_eur)
    bars_fx = [
        (50.0, 50.5, 49.98, 50.0),
        (49.9, 50.2, 49.5, 50.0),
    ]
    res_fx = _T212_REPLAY.run_replay(
        bars_fx, params_eur, bar_minutes=60.0, fx_to_usd=[1.0, 0.98], initial_cash=100.0,
    )
    assert res_fx["fills"] == 1
    assert res_fx["open_qty"] > 0.0
    assert res_fx["min_cash"] >= 0.0

    # scale_subbar_timeout_params scales sub-minute check_minutes timeouts proportionally
    scaled = spot_replay.scale_subbar_timeout_params(params, 60.0)
    assert scaled.tp_stagnation_max_sec > params.tp_stagnation_max_sec
    assert scaled.sl_trend_defer_max_sec > params.sl_trend_defer_max_sec


def test_backtest_cli_and_promotion_gate_validation(tmp_path: Path) -> None:
    """Verify negative/zero budget & malformed report handling in backtest CLIs and runners."""
    with pytest.raises(ValueError, match="invalid value"):
        _KRAKEN_BT._bt_params({"budget": -100})
    with pytest.raises(ValueError, match="budget"):
        _T212_BT.simulate([(100.0, 101.0, 99.0, 100.0)] * 25, {"budget": 0})

    bad_base = tmp_path / "bad_base.json"
    bad_cand = tmp_path / "bad_cand.json"
    atomic_write_json(bad_base, {"scenarios": {}}, indent=2)
    atomic_write_json(bad_cand, {"scenarios": {}}, indent=2)
    with pytest.raises(SystemExit):
        financial_promotion_gate.main([str(bad_base), str(bad_cand)])

    bad_wf = tmp_path / "bad_wf.json"
    atomic_write_json(
        bad_wf,
        {
            "purpose": "kraken baseline",
            "datasets": {},
            "strategy_params": {},
            "execution_model": {"fill_mode": "invalid_mode"},
        },
        indent=2,
    )
    with pytest.raises(SystemExit):
        walk_forward_compare.main([str(bad_wf)])


def test_wave8_2_slice_grid_ablation_and_sweep_invariants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify streaming slice normalization, T212 grid isolation, ablation unmasked Test 2, and sweep grid."""
    # 1. slice_dense_archive streams via iter_jsonl_dicts and normalizes symbol alias 's'
    fake_cache = tmp_path / "cachedb"
    prices_dir = fake_cache / "prices"
    prices_dir.mkdir(parents=True)
    alias_src = prices_dir / "cache_365_price_HYPEUSDC.jsonl"
    atomic_write_jsonl(
        alias_src,
        [
            {"s": "HYPEUSDC", "i": [1_700_000_000_000, 20.0]},
            {"s": "HYPEUSDC", "i": [1_700_200_000_000, 22.0]},
        ],
    )
    out_slice = tmp_path / "sliced_hype.jsonl"
    monkeypatch.setattr(sda, "cache_path", lambda fname: str(prices_dir / fname))
    sda.main(["HYPEUSD", "1", str(out_slice)])
    streamed = list(iter_jsonl_dicts(str(out_slice)))
    assert len(streamed) == 1
    assert streamed[0]["s"] == "HYPEUSD"
    assert load_price_series(str(out_slice), "HYPEUSD") == [(1_700_200_000.0, 22.0)]

    # 2. t212_continuous_grid disables reentry_ttl_hours on non-hybrid variants and report.g rejects bool
    p_spcx_re5, _ = t212_grid.params_for("spcx", "reentry5")
    assert p_spcx_re5.reentry_hybrid_enabled is False
    assert p_spcx_re5.reentry_ttl_hours == 0.0
    assert t212_report.g({"1h": {"cyc": True}}, "1h", "cyc") is None
    assert t212_report.g({"1h": {"cyc": 4}}, "1h", "cyc") == 4

    # 3. hybrid_reentry.harness disables reentry_pullback_adaptive so Test 2 != Test 3, and sorts CSVs
    base_adapt = dataclasses.replace(css.LIVE_PARAMS, reentry_pullback_adaptive=True)
    ab_vars = hr_harness.build_ablation_variants(base_adapt)
    assert ab_vars["Test 2"][1].reentry_pullback_adaptive is False
    assert ab_vars["Test 2"][1].reentry_pullback_pct == 0.0
    assert ab_vars["Test 3"][1].reentry_pullback_pct == 1.5
    assert ab_vars["Test 2"][1] != ab_vars["Test 3"][1]

    unsorted_csv = tmp_path / "unsorted.csv"
    atomic_write_text(
        str(unsorted_csv),
        "timestamp,open,high,low,close\n"
        "1700007200,12,13,11,12.5\n"
        "1700003600,10,11,9,10.5\n"
        "1700003600,10,12,9,11.0\n",
    )
    rows, ts_list = hr_harness.load_ohlc_with_timestamps_csv(str(unsorted_csv))
    assert ts_list == [1700003600, 1700007200]
    assert rows[0][3] == 11.0 and rows[1][3] == 12.5

    # 4. comprehensive_strategy_sweep unmasks fixed-TP and static re-entry candidates
    cgrid = css.define_candidate_grid()
    assert cgrid["01_tp4"]["tp_trend_hold"] is False
    assert cgrid["15_reentry_pullback_15"]["reentry_ttl_hours"] == 0.0
    assert cgrid["15_reentry_pullback_15"]["reentry_pullback_adaptive"] is False


def test_wave8_2_monitortrades_regime_pilot_and_ml_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 900s gap handling, symbol fee/quote SSOT, ms timestamp normalization, pilot rate-limit, and ML guards."""
    # 1. evaluate_regime_900s does not look across multi-day archive gaps when [t0, t0+900s] has no sample
    gap_ts = np.array([1000.0, 100_000.0, 100_100.0], dtype=np.float64)
    gap_px = np.array([100.0, 200.0, 201.0], dtype=np.float64)
    sig_gap = bwr.compute_exact_time_window_signals(gap_ts, gap_px, 900.0, "BTCUSDC")
    res_gap, _, _ = bwr.evaluate_regime_900s(gap_ts, gap_px, sig_gap, horizon_sec=900.0, stride=1)
    assert res_gap == []

    # 2. run_replay_backtest resolves Kraken USD/0.26% vs Binance USDC/0.10%
    assert rrb._resolve_symbol_meta("HYPEUSD") == ("USD", 0.26)
    assert rrb._resolve_symbol_meta("BTCUSDC") == ("USDC", 0.1)

    # 3. ReplayTrendSource normalizes millisecond timestamps to seconds
    rts_src = rts.ReplayTrendSource(["BTCUSDC"], window_seconds=90.0)
    for idx in range(5):
        rts_src.advance("BTCUSDC", (1_700_000_000 + idx * 10) * 1000, 100.0 + idx)
    assert rts_src._prev_ts["BTCUSDC"] == pytest.approx(1_700_000_040.0)
    assert rts_src.is_trend_up("BTCUSDC", window_seconds=60.0) is True

    # 4. scheduled_pilot.is_rate_limited does not fail open when latest applied row has corrupt timestamp
    audit_path = tmp_path / "pilot_audit.jsonl"
    recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
    atomic_write_jsonl(
        audit_path,
        [
            {"full_key": "BINANCE_BTC.mt.gain", "action": "applied", "ts": recent_ts},
            {"full_key": "BINANCE_BTC.mt.gain", "action": "applied", "ts": "not-a-timestamp"},
        ],
    )
    monkeypatch.setattr(sp, "AUDIT_LOG", str(audit_path))
    limited, lim_ts = sp.is_rate_limited("BINANCE_BTC.mt.gain")
    assert limited is True
    assert lim_ts == recent_ts

    # 5. scheduled_pilot._apply_config_change rolls back invalid writes & evaluate_key breaks ties toward current
    conf_file = tmp_path / "instruments.conf"
    atomic_write_text(conf_file, "[BTCEUR]\nmt.gain = 7.5 ; inline comment\n")
    monkeypatch.setattr(sp, "INSTRUMENTS_CONF", str(conf_file))
    sp._apply_config_change("BTCEUR", "mt.gain", 7.5, 8.25)
    assert sp._current_value("BTCEUR", "mt.gain") == pytest.approx(8.25)
    with pytest.raises(ValueError):
        sp._apply_config_change("BTCEUR", "mt.gain", 8.25, "bad_num")
    assert "mt.gain = 8.25" in conf_file.read_text(encoding="utf-8")

    monkeypatch.setattr(sp, "scan_backtest_ranges", lambda _p: {"BINANCE_OK.mt.gain": ["4.0", "6.0", "8.0"]})
    monkeypatch.setattr(sp, "_current_value", lambda _s, _k: 6.0)
    monkeypatch.setattr(sp.rb, "SYMBOLS", {"OKUSDC": {"params": {"mt.gain": "6.0"}}})
    monkeypatch.setattr(sp, "_split_series", lambda _sym: ([1] * 120, [1] * 120))
    monkeypatch.setattr(sp, "_run_one", lambda *_a, **_kw: {"net": 50.0, "buy_hold": 0.0})
    tie_res = sp.evaluate_key("BINANCE_OK.mt.gain", "OKUSDC", "OK", "mt.gain", dry_run=True, propose=True)
    assert tie_res["action"] == "no_change"
    assert tie_res["winner_half1"] == "6.0"

    # 6. ml_forecast.build_dataset skips windows spanning timestamp gaps; PricePrediction caps max_history
    px_ml = np.linspace(100.0, 130.0, ml_fc.WARMUP + ml_fc.HORIZON_H + 5)
    ts_gapped = np.arange(len(px_ml), dtype=np.float64) * 3600.0
    ts_gapped[ml_fc.WARMUP + 2 :] += 10 * 86400.0
    X_gap, _, _ = ml_fc.build_dataset(px_ml, ts=ts_gapped)
    assert len(X_gap) == 0

    pp = PricePrediction(window_size=4, epochs=2, max_history=12, retrain_every=5)
    for i in range(20):
        pp.process_price(100.0 + i)
    assert len(pp.prices) == 12

    # 7. vol_chronos flat volatility (actual == base, pred > base) scores 0.0 direction accuracy (not 1.0)
    monkeypatch.setattr(ml_vc, "_load_pipeline", lambda: make_fake_chronos_pipeline(0.5))
    monkeypatch.setattr(ml_vc, "torch", SimpleNamespace(float32="f32", tensor=lambda a, dtype=None: np.asarray(a)))
    flat_rv = np.full(30, 0.2, dtype=np.float64)
    wf_flat = ml_vc.walk_forward_vol(flat_rv, horizon=5, warmup=5, stride=10)
    assert wf_flat["direction_accuracy"] == 0.0
