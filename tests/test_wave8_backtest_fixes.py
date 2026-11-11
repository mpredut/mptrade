"""Targeted regression tests for Wave 8.1 backtest, replay, strategy, and runner fixes."""

from __future__ import annotations

import ast
import dataclasses
import subprocess
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from botcore import load_local_module
from offline.backtests.evaluation import load_venue_replay_module
from offline.runners import financial_promotion_gate, walk_forward_compare
from state_io import atomic_write_json, read_text_file
from strategies import spot_engine, spot_replay, spot_rules
from strategies.base_spot_mixin import BaseSpotMixin, sanitize_common_spot_state
from strategies.state_store import (
    compute_adaptive_reentry_pullback_pct,
    safe_state_token,
    update_post_sell_anchors,
)

ROOT = Path(__file__).resolve().parents[1]
_KRAKEN_BT = load_local_module(str(ROOT / "kraken"), "backtest", module_key="test_w8_kraken_bt")
_T212_BT = load_local_module(str(ROOT / "212trading"), "backtest", module_key="test_w8_t212_bt")
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
        load_local_module(str(ROOT / "kraken"), "kraken_common", module_key="kraken_common")
        .load_isolated_kraken_cfg(ROOT / "kraken")
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
