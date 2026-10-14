from __future__ import annotations

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HL_DIR = os.path.join(_ROOT, "hyperliquid")
if _HL_DIR not in sys.path:
    sys.path.insert(0, _HL_DIR)

spec = importlib.util.spec_from_file_location("hl_backtest_module", os.path.join(_HL_DIR, "backtest.py"))
hl_backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hl_backtest)
simulate = hl_backtest.simulate

spec_replay = importlib.util.spec_from_file_location("hl_replay_module", os.path.join(_HL_DIR, "replay.py"))
hl_replay = importlib.util.module_from_spec(spec_replay)
spec_replay.loader.exec_module(hl_replay)
make_hyperliquid_replay_adapter = hl_replay.make_hyperliquid_replay_adapter
run_replay = hl_replay.run_replay

from kraken.backtest_trail import simulate_trail


def test_hyperliquid_simulate_stop_loss_long():
    # Price enters at 99.8, drops past 5% SL to 90.0
    ohlc = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 100.5, 99.5, 99.8),  # hits entry discount at 99.8
        (99.8, 100.0, 95.0, 96.0),   # drops
        (96.0, 96.0, 89.0, 90.0),    # drops below -5% SL (loss is ~9.8%)
    ]
    ts = [1000, 2000, 3000, 4000]
    fund = [(0, 0.0001)]
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=0, budget=500.0, fee=0.045, sl=5.0)

    m = simulate(ohlc, ts, fund, sig, P, "long")
    assert m["cycles"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] < 0  # Stop loss realized loss


def test_hyperliquid_simulate_stop_loss_short():
    # Short enters at 100.2 (100 * (1 - (-1)*0.002) = 100.2)
    # Price rises past 5% SL to 110.0
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.2),  # fills short entry at 100.2
        (100.2, 105.0, 100.0, 104.0),
        (104.0, 111.0, 103.0, 110.0), # rises past 5% SL on short
    ]
    ts = [1000, 2000, 3000, 4000]
    fund = [(0, 0.0001)]
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=0, budget=500.0, fee=0.045, sl=5.0)

    m = simulate(ohlc, ts, fund, sig, P, "short")
    assert m["cycles"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] < 0


def test_kraken_simulate_trail_high_peak():
    # Tests that trailing exit arms on intrabar high even if close is lower
    # Entry discount 0.8% on close 100 -> buy at 99.2
    # Bar 2: Low 99.0 fills buy at 99.2. Close 100.
    # Bar 3: High 106.0 reaches +5% TP (99.2 * 1.05 = 104.16), arms with peak 106.0.
    #        Close at 101.0 (which is <= 106.0 * (1 - 0.03) = 102.82) -> triggers trailing TP!
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.0, 100.0),
        (100.0, 106.0, 100.0, 101.0),
    ]
    P = dict(entry=300, dca=150, disc=0.8, drop=1.0, tp=5.0, maxdca=10,
             budget=3000, fee=0.25, sl=7.0, trail=3.0)

    m = simulate_trail(ohlc, P)
    assert m["cycles"] == 1
    assert m["wins"] == 1
    assert m["open_qty"] == 0.0


def test_hyperliquid_simulate_take_profit_long():
    # Bar 0: c=100.0 -> entry limit buy at 99.8 (0.2% discount)
    # Bar 1: low=99.5 -> fills buy at 99.8. Close=100.0 -> TP limit sell at 99.8 * 1.01 = 100.798
    # Bar 2: high=102.0 -> fills TP at 100.798 -> profitable cycle complete
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 102.0, 100.0, 101.5),
    ]
    ts = [1000, 2000, 3000]
    fund = [(0, 0.0)]
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0)

    m = simulate(ohlc, ts, fund, sig, P, "long")
    assert m["cycles"] == 1
    assert m["wins"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] > 0.0


def test_hyperliquid_simulate_take_profit_short():
    # Bar 0: c=100.0 -> short entry limit sell at 100.2 (0.2% discount)
    # Bar 1: high=100.5 -> fills short entry at 100.2. Close=100.0 -> TP limit buy at 100.2 * 0.99 = 99.198
    # Bar 2: low=98.5 -> fills TP buy at 99.198 -> profitable cycle complete
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 98.5, 99.0),
    ]
    ts = [1000, 2000, 3000]
    fund = [(0, 0.0)]
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0)

    m = simulate(ohlc, ts, fund, sig, P, "short")
    assert m["cycles"] == 1
    assert m["wins"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] > 0.0


def test_hyperliquid_simulate_dca_execution():
    # Bar 0: c=100.0 -> buy at 99.8
    # Bar 1: low=99.5 -> fills entry at 99.8
    # Bar 2: price drops to 97.0 (> 2% drop from 99.8) -> DCA order placed at 97.0 * (1 - 0.002) = 96.806
    # Bar 3: low=96.5 -> fills DCA buy. New avg is ~98.3. New TP placed at ~99.28.
    # Bar 4: high=101.0 -> fills TP sell!
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 99.8),
        (99.8, 100.0, 97.0, 97.0),
        (97.0, 97.0, 96.5, 96.8),
        (96.8, 101.0, 96.8, 100.0),
    ]
    ts = [1000, 2000, 3000, 4000, 5000]
    fund = [(0, 0.0)]
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0)

    m = simulate(ohlc, ts, fund, sig, P, "long")
    assert m["cycles"] == 1
    assert m["wins"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] > 0.0


def test_hyperliquid_simulate_signal_gate():
    # Signal is 'down' on bars 0 and 1 -> long entry gate blocks placement
    # Signal becomes 'up' on bar 2 -> entry order placed
    # Bar 3 low fills entry
    ohlc = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ]
    ts = [1000, 2000, 3000, 4000]
    fund = [(0, 0.0)]
    sig = ["down", "down", "up", "up"]
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=0, budget=500.0, fee=0.045, sl=5.0, signal_gate=True)

    m = simulate(ohlc, ts, fund, sig, P, "long")
    # By bar 3, order was placed on bar 2 and filled on bar 3
    assert m["open_qty"] > 0.0


def test_hyperliquid_simulate_funding():
    # Short position receiving positive funding vs long paying it
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ]
    ts = [1000, 2000, 3000]
    fund = [(0, 0.001)]  # 0.1% positive funding rate
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.0, drop=2.0, tp=10.0, maxdca=0, budget=500.0, fee=0.045, sl=0.0)

    m_long = simulate(ohlc, ts, fund, sig, P, "long")
    m_short = simulate(ohlc, ts, fund, sig, P, "short")

    assert m_long["funding"] < 0.0    # long pays funding
    assert m_short["funding"] > 0.0   # short receives funding


def test_hyperliquid_standardized_performance_metrics():
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 102.0, 100.0, 101.5),
    ]
    ts = [1000, 2000, 3000]
    fund = [(0, 0.0)]
    sig = ["__any__"] * len(ohlc)
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0)

    m = simulate(ohlc, ts, fund, sig, P, "long")

    assert "return_pct" in m
    assert "max_drawdown_pct" in m
    assert "max_drawdown_abs" in m
    assert "exposure_pct" in m
    assert "turnover_pct" in m
    assert "win_rate_pct" in m
    assert "net_pnl" in m
    assert "fills" in m
    assert "initial_capital" in m
    assert m["initial_capital"] == 500.0
    assert m["fills"] >= 2
    assert m["win_rate_pct"] == 100.0
    assert m["return_pct"] > 0.0
    assert m["trade_count"] == 1
    assert m["exposure_pct"] > 0.0


def test_hyperliquid_walk_forward_adapter_compatibility():
    from offline.backtests.evaluation import evaluate_segment

    records = [
        {"timestamp": 1000 + i * 3600, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}
        for i in range(10)
    ]
    # Price moves to fill entry and TP
    records[1] = {"timestamp": 1000 + 1 * 3600, "open": 100.0, "high": 100.5, "low": 99.0, "close": 99.8}
    records[2] = {"timestamp": 1000 + 2 * 3600, "open": 99.8, "high": 102.0, "low": 99.5, "close": 101.5}

    adapter = make_hyperliquid_replay_adapter(
        params=dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0),
        direction="long",
        coin="HYPE",
        bar_minutes=60.0,
    )
    res = evaluate_segment(records, adapter)
    assert "metrics" in res
    assert "return_pct" in res["metrics"]
    assert "max_drawdown_pct" in res["metrics"]
    assert "cycles" in res["metrics"]
    assert "fills" in res["metrics"]
    assert res["metrics"]["cycles"] == 1
    assert res["metrics"]["fills"] >= 2
    assert res["buy_hold_return_pct"] is not None


def test_hyperliquid_order_ttl_runaway_cancellation():
    # Bar 0: price = 100. Entry order placed at 100 * (1 - 0.002) = 99.8
    # Bar 1: price runs away upwards to 105. Since 105 > 99.8 * 1.003 and order_ttl_min (30m) < bar_minutes (60m),
    # the stale resting entry order is cancelled and replaced at current market price 105 * 0.998.
    ohlc = [
        (100.0, 100.2, 99.9, 100.0),
        (101.0, 105.0, 100.5, 104.5),
        (104.5, 104.8, 103.5, 104.0),
    ]
    ts = [1000, 1000 + 3600, 1000 + 7200]
    res = run_replay(
        ohlc,
        ts=ts,
        params=dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0, order_ttl_min=30.0),
        direction="long",
        coin="HYPE",
        bar_minutes=60.0,
    )
    assert res is not None
    assert "return_pct" in res


def test_hyperliquid_intrabar_stop_loss():
    # Bar 0: places entry at 99.8
    # Bar 1: fills entry at 99.8 (position is open with avg ~99.8)
    # Bar 2: plunges intrabar to low 85.0 (-14.8%), breaching 10% SL, but closes at 98.0 (-1.8%)
    ohlc = [
        (100.0, 100.2, 99.5, 100.0),
        (100.0, 100.2, 99.5, 99.8),
        (99.8, 100.0, 85.0, 98.0),
    ]
    ts = [1000, 2000, 3000]
    res = run_replay(
        ohlc,
        ts=ts,
        params=dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=5.0, maxdca=2, budget=500.0, fee=0.045, sl=10.0),
        direction="long",
        coin="HYPE",
        bar_minutes=60.0,
    )
    # Intrabar stop-loss must have triggered (cycles >= 1)
    assert res["cycles"] >= 1
    assert res["realized"] < 0  # Took a loss on stop-loss
    assert res["fills"] >= 2


def test_hyperliquid_ambiguous_bars_and_execution_model():
    from offline.backtests.execution import ExecutionModel

    # Bar 0: places entry at 99.8
    # Bar 1: fills entry at 99.8, places TP at ~101.8
    # Bar 2: drop to 97.5 triggers DCA placement at ~97.3
    # Bar 3: wide candle [96.5, 102.5] touches both resting DCA (open) and resting TP (close)
    ohlc = [
        (100.0, 100.2, 99.5, 100.0),
        (100.0, 100.2, 99.5, 99.8),
        (99.8, 100.0, 97.5, 97.5),
        (97.5, 102.5, 96.5, 100.0),
    ]
    ts = [1000, 2000, 3000, 4000]
    res_worst = run_replay(
        ohlc,
        ts=ts,
        params=dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=2.0, maxdca=2, budget=500.0, fee=0.045, sl=15.0),
        direction="long",
        coin="HYPE",
        execution=ExecutionModel(intrabar_policy="worst_case"),
    )
    assert res_worst["ambiguous_bars"] >= 1
    assert "intrabar_policy_selected" in res_worst
    assert "intrabar_scenarios" in res_worst


def test_validate_replay_metrics_across_all_venues():
    from offline.backtests.metrics import validate_replay_metrics
    from strategies.spot_engine import StratParams as KrakenStratParams
    import kraken.replay as kr_replay
    import importlib.util

    t212_spec = importlib.util.spec_from_file_location(
        "t212_replay_validation_test",
        os.path.join(_ROOT, "212trading", "replay.py"),
    )
    t212_mod = importlib.util.module_from_spec(t212_spec)
    sys.modules[t212_spec.name] = t212_mod
    t212_spec.loader.exec_module(t212_mod)

    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 99.0, 101.0),
    ]

    hl_res = run_replay(
        ohlc,
        params=dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=1.0, maxdca=2, budget=500.0, fee=0.045, sl=5.0),
        direction="long",
        coin="HYPE",
    )
    validate_replay_metrics(hl_res)

    kr_res = kr_replay.run_replay(
        ohlc,
        KrakenStratParams(
            currency="USD",
            entry_amount=100.0,
            entry_discount_pct=0.2,
            dca_amount=50.0,
            dca_drop_pct=2.0,
            takeprofit_pct=1.0,
            max_budget=500.0,
            max_dca_buys=2,
            check_minutes=1.0,
            enable_takeprofit=True,
            order_ttl_min=60.0,
            stop_loss_pct=5.0,
        ),
        fee_pct=0.25,
    )
    validate_replay_metrics(kr_res)

    t212_res = t212_mod.run_replay(
        ohlc,
        t212_mod._strat.StratParams(
            currency="USD",
            entry_amount=100.0,
            entry_discount_pct=0.2,
            dca_amount=50.0,
            dca_drop_pct=2.0,
            check_minutes=5.0,
            takeprofit_pct=1.0,
            max_budget=500.0,
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
        ),
    )
    validate_replay_metrics(t212_res)



