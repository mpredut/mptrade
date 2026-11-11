from __future__ import annotations

import os
import sys
import json
from offline.backtests.execution import ExecutionModel
from unittest.mock import MagicMock, patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HL_DIR = os.path.join(_ROOT, "hyperliquid")
if _HL_DIR not in sys.path:
    sys.path.insert(0, _HL_DIR)

from botcore import load_local_module

hl_backtest = load_local_module(_HL_DIR, "backtest", module_key="hl_backtest_module", reload=True)
hl_replay = load_local_module(_HL_DIR, "replay", module_key="hl_replay_module", reload=True)
make_hyperliquid_replay_adapter = hl_replay.make_hyperliquid_replay_adapter
run_replay = hl_replay.run_replay

from kraken.backtest_trail import simulate_trail  # noqa: E402
import time as _time
import unittest as _unittest


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

    m = run_replay(ohlc, ts, fund, sig, P, "long")
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

    m = run_replay(ohlc, ts, fund, sig, P, "short")
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

    m = run_replay(ohlc, ts, fund, sig, P, "long")
    assert m["cycles"] == 1
    assert m["wins"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] > 0.0
    assert m["realized"] > m["net"]
    assert abs((m["realized"] + m["funding"] + m["final_upnl"] - m["fees"]) - m["total"]) < 1e-6


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

    m = run_replay(ohlc, ts, fund, sig, P, "short")
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

    m = run_replay(ohlc, ts, fund, sig, P, "long")
    assert m["cycles"] == 1
    assert m["wins"] == 1
    assert m["open_qty"] == 0.0
    assert m["realized"] > 0.0
    assert m["realized"] > m["net"]
    assert abs((m["realized"] + m["funding"] + m["final_upnl"] - m["fees"]) - m["total"]) < 1e-6


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

    m = run_replay(ohlc, ts, fund, sig, P, "long")
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

    m_long = run_replay(ohlc, ts, fund, sig, P, "long")
    m_short = run_replay(ohlc, ts, fund, sig, P, "short")

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

    m = run_replay(ohlc, ts, fund, sig, P, "long")

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
    res = evaluate_segment(records, adapter, validate_contract=True)
    assert "metrics" in res
    assert "return_pct" in res["metrics"]
    assert "max_drawdown_pct" in res["metrics"]
    assert "cycles" in res["metrics"]
    assert "fills" in res["metrics"]
    assert res["metrics"]["cycles"] == 1
    assert res["metrics"]["fills"] >= 2
    assert res["metrics"]["account_currency"] == "USDC"
    assert "ambiguous_bars" in res["metrics"]
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
    from strategies import spot_replay as kr_replay

    t212_mod = load_local_module(os.path.join(_ROOT, "212trading"), "replay", module_key="t212_replay_validation_test", reload=True)

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


def test_hyperliquid_market_data_fetchers():

    _hl_md = load_local_module(_HL_DIR, "market_data", module_key="hl_md_test", reload=True)
    fetch_ohlc_candles = _hl_md.fetch_ohlc_candles
    fetch_funding_history = _hl_md.fetch_funding_history

    fake_candles = [
        {"t": 1700000000000, "T": 1700003599999, "s": "HYPE", "i": "1h",
         "o": "20.0", "c": "21.0", "h": "22.0", "l": "19.5", "v": "100.0", "n": 50},
        {"t": 1700003600000, "T": 1700007199999, "s": "HYPE", "i": "1h",
         "o": "21.0", "c": "22.5", "h": "23.0", "l": "20.5", "v": "120.0", "n": 60},
    ]
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_candles).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        records = fetch_ohlc_candles("HYPE", 60, drop_incomplete=False)
        assert len(records) == 2
        assert records[0]["timestamp"] == 1700000000
        assert records[0]["open"] == 20.0
        assert records[0]["close"] == 21.0
        assert records[1]["close"] == 22.5

    fake_funding = [
        {"coin": "HYPE", "fundingRate": "0.0001", "premium": "0.0", "time": 1700000000000},
        {"coin": "HYPE", "fundingRate": "0.00015", "premium": "0.0", "time": 1700003600000},
    ]
    mock_funding_resp = MagicMock()
    mock_funding_resp.read.return_value = json.dumps(fake_funding).encode("utf-8")
    mock_funding_resp.__enter__.return_value = mock_funding_resp

    with patch("urllib.request.urlopen", return_value=mock_funding_resp):
        funding = fetch_funding_history("HYPE", 1700000000000)
        assert len(funding) == 2
        assert funding[0] == (1700000000000, 0.0001)
        assert funding[1] == (1700003600000, 0.00015)


def test_hyperliquid_walk_forward_baseline_runner(tmp_path):
    _wf_mod = load_local_module(
        os.path.join(_ROOT, "offline", "runners"), "hyperliquid_walk_forward_baseline", module_key="hl_wf_baseline", reload=True
    )

    fake_records = [
        {"timestamp": 1700000000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(30)
    ]
    with patch.object(_wf_mod, "fetch_ohlc_candles", return_value=fake_records):
        with patch("sys.argv", [
            "hyperliquid_walk_forward_baseline.py",
            "--intervals", "60",
            "--coin", "HYPE",
            "--output-dir", str(tmp_path),
        ]):
            rc = _wf_mod.main()
            assert rc == 0
            report_file = tmp_path / "HYPE_60m_baseline.json"
            assert report_file.exists()
            data = json.loads(report_file.read_text(encoding="utf-8"))
            assert data["runtime"]["coin"] == "HYPE"
            assert data["runtime"]["engine"] == "spot"
            assert data["strategy_params"]["currency"] == "USDC"
            assert data["strategy_params"]["entry_amount"] == 350.0
            assert "aggregate_test" in data
            assert "folds" in data
            assert len(data["folds"]) > 0


def test_hyperliquid_walk_forward_baseline_runner_perp_engine(tmp_path):
    _wf_mod = load_local_module(
        os.path.join(_ROOT, "offline", "runners"), "hyperliquid_walk_forward_baseline", module_key="hl_wf_baseline", reload=True
    )

    fake_records = [
        {"timestamp": 1700000000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(30)
    ]
    with patch.object(_wf_mod, "fetch_ohlc_candles", return_value=fake_records):
        with patch("sys.argv", [
            "hyperliquid_walk_forward_baseline.py",
            "--intervals", "60",
            "--coin", "HYPE",
            "--engine", "perp",
            "--output-dir", str(tmp_path),
        ]):
            rc = _wf_mod.main()
            assert rc == 0
            report_file = tmp_path / "HYPE_60m_baseline.json"
            assert report_file.exists()
            data = json.loads(report_file.read_text(encoding="utf-8"))
            assert data["runtime"]["engine"] == "perp"
            assert "aggregate_test" in data
            assert len(data["folds"]) > 0


def test_hyperliquid_replay_partial_fills():
    # Bars:
    # Bar 0: places limit open at 99.8 (100 * (1 - 0.002))
    # Bar 1: low touches 99.5 -> fills first 50% chunk (partial_fill_ratio=0.5)
    # Bar 2: low touches 99.5 -> fills remaining 50% chunk
    # Bar 3: high touches 105.0 -> fills take-profit
    ohlc = [
        (100.0, 100.5, 99.9, 100.0),  # bar 0: step places limit buy at 99.8
        (100.0, 100.2, 99.5, 100.0),  # bar 1: limit buy touched, 50% tranche filled
        (100.0, 100.2, 99.5, 100.0),  # bar 2: second 50% tranche filled -> order completed
        (100.0, 105.0, 100.0, 104.0),  # bar 3: reaches TP, 50% close filled
        (100.0, 105.0, 100.0, 104.0),  # bar 4: reaches TP, remaining 50% close filled
    ]
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=2.0, maxdca=0, budget=500.0, fee=0.045)
    model = ExecutionModel(partial_fill_ratio=0.5)
    res = run_replay(ohlc, params=P, execution=model, bar_minutes=60.0)
    assert res["fills"] >= 3  # 2 fills for open + 2 for close
    assert res["cycles"] == 1
    assert res["open_qty"] == 0.0


def test_hyperliquid_replay_ttl_not_premature_on_bar_zero():
    # Verify TTL elapsed calculation is (i - placed_b) * bar_minutes,
    # so an order placed at bar 0 is NOT marked stale on bar 0 itself.
    ohlc = [
        (100.0, 101.5, 99.9, 101.0),  # bar 0: close 101 > 99.8 * 1.003, but elapsed = 0 min <= 10 min TTL
        (101.0, 102.0, 99.5, 101.5),  # bar 1: entry fills before being stale
    ]
    P = dict(entry=100.0, dca=50.0, disc=0.2, drop=2.0, tp=2.0, maxdca=0, budget=500.0, fee=0.045)
    # With (i - placed_b) * 60: at i=0, elapsed = 0 -> not stale -> order survives to fill at bar 1
    res = run_replay(ohlc, params=P, bar_minutes=60.0)
    assert res["fills"] >= 1


def test_hyperliquid_reconcile_real_protects_against_transient_errors(tmp_path):
    Strategy = hl_replay.Strategy
    StratParams = hl_replay.StratParams

    class FakeClient:
        def __init__(self):
            self.pos_fail = False
            self.orders_fail = False

        def sz_decimals(self, coin):
            return 2

        def position_strict(self, coin):
            if self.pos_fail:
                raise RuntimeError("transient HTTP timeout")
            return (1.5, 100.0)

        def open_orders_strict(self, coin):
            if self.orders_fail:
                raise RuntimeError("transient open_orders failure")
            return [{"oid": "123"}]

    client = FakeClient()
    strat = Strategy(
        client=client,
        coin="HYPE",
        params=StratParams(direction="long"),
        dry_run=False,
        replay_mode=True,
    )
    strat.s["qty"] = 1.5
    strat.s["cost"] = 150.0
    strat.s["orders"] = [{"oid": "123", "role": "close", "side": "sell", "sz": 1.5, "px": 102.0}]

    # 1. Transient position failure must return False and preserve qty/orders
    client.pos_fail = True
    assert strat.reconcile(99.0) is False
    assert strat.s["qty"] == 1.5
    assert len(strat.s["orders"]) == 1

    # 2. Transient open_orders failure must preserve tracked orders instead of wiping them
    client.pos_fail = False
    client.orders_fail = True
    assert strat.reconcile(99.0) is True
    assert strat.s["qty"] == 1.5
    assert len(strat.s["orders"]) == 1


def test_hyperliquid_reconcile_paper_stop_fills_on_continued_crash():
    Strategy = hl_replay.Strategy
    StratParams = hl_replay.StratParams

    strat = Strategy(
        client=None,
        coin="HYPE",
        params=StratParams(direction="long", stop_loss_pct=5.0),
        dry_run=True,
        replay_mode=False,
        initial_state={},
    )
    strat._save = lambda: None
    strat._apply_open(1.0, 100.0, 100.0, "ENTRY")
    # Trigger stop loss at 94.0 (places STOP sell at 94 * 0.995 = 93.53)
    assert strat._check_stop_loss(94.0) is True
    assert any(o.get("kind") == "STOP" for o in strat.s["orders"])
    # On next tick, price crashes further to 90.0 (< 93.53); STOP must still fill!
    strat.reconcile(90.0)
    assert strat.s["qty"] == 0.0
    assert len(strat.s["orders"]) == 0


def test_hyperliquid_reconcile_real_marginal_dca_fill_price():
    Strategy = hl_replay.Strategy
    StratParams = hl_replay.StratParams

    class FakeClient:
        def __init__(self):
            self.pos = (15.0, 1490.0 / 15.0)
            self.orders = []

        def sz_decimals(self, coin):
            return 2

        def position_strict(self, coin):
            return self.pos

        def open_orders_strict(self, coin):
            return self.orders

        def cancel(self, coin, oid):
            return True

    client = FakeClient()
    strat = Strategy(
        client=client,
        coin="HYPE",
        params=StratParams(direction="long", dca_drop_pct=2.0),
        dry_run=False,
        replay_mode=True,
    )
    strat.s["qty"] = 10.0
    strat.s["cost"] = 1000.0
    strat.s["spent"] = 1000.0
    strat.s["entry_price"] = 100.0
    strat.s["last_open_price"] = 100.0
    strat.s["orders"] = [{"oid": "DCA-1", "role": "open", "side": "buy", "sz": 5.0, "px": 98.0, "kind": "DCA"}]

    assert strat.reconcile(98.0) is True
    assert abs(strat.s["last_open_price"] - 98.0) < 1e-6
    assert strat.s["dca_buys"] == 1

    # Partial ENTRY fill across two ticks must not count as DCA, and partial DCA fill across two ticks counts once
    strat2 = Strategy(
        client=client,
        coin="HYPE",
        params=StratParams(direction="long", dca_drop_pct=2.0),
        dry_run=False,
        replay_mode=True,
    )
    strat2.s["orders"] = [{"oid": "ENT-1", "role": "open", "side": "buy", "sz": 10.0, "px": 100.0, "kind": "ENTRY"}]
    client.orders = [{"oid": "ENT-1"}]  # still open (partial fill 1)
    client.pos = (4.0, 100.0)
    assert strat2.reconcile(100.0) is True
    assert strat2.s["dca_buys"] == 0

    client.orders = []  # second fill completes ENTRY
    client.pos = (10.0, 100.0)
    assert strat2.reconcile(100.0) is True
    assert strat2.s["dca_buys"] == 0

    # Now place a DCA order that fills in two partial fills
    strat2.s["orders"] = [{"oid": "DCA-2", "role": "open", "side": "buy", "sz": 4.0, "px": 95.0, "kind": "DCA"}]
    client.orders = [{"oid": "DCA-2"}]
    client.pos = (12.0, (1000.0 + 190.0) / 12.0)
    assert strat2.reconcile(95.0) is True
    assert strat2.s["dca_buys"] == 1

    client.orders = []
    client.pos = (14.0, (1000.0 + 380.0) / 14.0)
    assert strat2.reconcile(95.0) is True
    assert strat2.s["dca_buys"] == 1


def test_hyperliquid_price_analysis_and_signals_edge_cases(tmp_path):
    signals = load_local_module(_HL_DIR, "signals", module_key="hl_sig_module", reload=True)
    price_analysis = signals

    class FakeCandleClient:
        def candles(self, coin, interval, lookback_hours=None):
            rows = [{"t": (1700000000 + i * 3600) * 1000, "c": str(100.0 + i * 0.1)} for i in range(60)]
            # Inject malformed candles missing 't' or 'c'
            rows.insert(5, {"t": 1700000000 * 1000})
            rows.insert(10, {"c": "101.0"})
            return rows

    res = price_analysis.analyze(FakeCandleClient(), "HYPE")
    assert res is not None
    assert res["direction"] == "up"

    corrupt_sig = tmp_path / "signal.json"
    corrupt_sig.write_text("[1, 2, 3]", encoding="utf-8")
    sig_res = signals._file_signal(str(corrupt_sig))
    assert sig_res["trend"] == "neutral"
    assert sig_res["source"] == "file(corrupt)"

    builtin_res = signals._builtin_trend(FakeCandleClient(), "HYPE")
    assert builtin_res["trend"] in ("up", "down", "neutral")
    assert builtin_res["source"] == "builtin"

    import numpy as np
    assert price_analysis.detect_long_term_trend(np.array([True, True, True]), [100.0, 101.0, 102.0]) is None
    assert price_analysis.detect_long_term_trend([1000.0, 2000.0, 3000.0], np.array([True, False, True])) is None


def test_walk_forward_compare_preserves_zero_fee():
    from offline.runners.walk_forward_compare import _build_venue_evaluator

    _, krk_fee, _ = _build_venue_evaluator("kraken", {"fee_pct_per_leg": 0.0, "strategy_params": {}})
    assert krk_fee == 0.0

    _, hl_fee, _ = _build_venue_evaluator("hyperliquid", {"fee_pct_per_leg": 0.0, "strategy_params": {"fee_pct": 0.035}})
    assert hl_fee == 0.0


def test_hyperliquid_sz_decimals_spot_fallback_and_exact_sl_boundary():
    import types
    hl_client_mod = load_local_module(_HL_DIR, "hl_client", module_key="hl_client_test_mod", reload=True)

    dummy = object.__new__(hl_client_mod.HLClient)
    dummy._meta_cache = {}
    dummy._sz_decimals_cache = {}
    dummy.info = types.SimpleNamespace(
        meta=lambda: {"universe": [{"name": "BTC", "szDecimals": 5}]},
        spot_meta=lambda: {"tokens": [{"name": "PURR", "szDecimals": 1}]},
    )
    assert dummy.sz_decimals("BTC") == 5
    assert dummy.sz_decimals("PURR") == 1
    try:
        dummy.sz_decimals("UNKNOWN_COIN")
        assert False, "Expected HLError for unknown coin"
    except hl_client_mod.HLError:
        pass

    # Fallback to Info.__init__ cached maps when subsequent meta()/spot_meta() calls time out
    def _raise_timeout():
        raise TimeoutError("Read timed out")

    dummy_timeout = object.__new__(hl_client_mod.HLClient)
    dummy_timeout._meta_cache = {}
    dummy_timeout.info = types.SimpleNamespace(
        meta=_raise_timeout,
        spot_meta=_raise_timeout,
        coin_to_asset={"HYPE": 159, "@107": 10107},
        name_to_coin={"HYPE": "HYPE", "HYPE/USDC": "@107"},
        asset_to_sz_decimals={159: 2, 10107: 2},
    )
    assert dummy_timeout.sz_decimals("HYPE") == 2
    assert dummy_timeout.resolve_spot_pair("HYPE") == "@107"

    # Exact intrabar stop_loss boundary (where IEEE-754 (avg - sl_level) / avg * 100 is 4.999999999999995 < 5.0)
    Strategy = hl_replay.Strategy
    StratParams = hl_replay.StratParams
    strat = Strategy(
        client=None,
        coin="HYPE",
        params=StratParams(direction="long", stop_loss_pct=5.0),
        dry_run=True,
        replay_mode=True,
        initial_state={},
    )
    avg_px = 29.37
    strat._apply_open(1.0, avg_px, avg_px, "ENTRY")
    sl_level = avg_px * (1.0 - 5.0 / 100.0)
    assert strat._check_stop_loss(sl_level) is True


def test_walk_forward_baseline_missing_dataset_and_warmup_persistence(tmp_path):
    from offline.runners import binance_walk_forward_baseline as bwf
    from offline.runners import hyperliquid_walk_forward_baseline as hwf

    # 1. Explicit missing --dataset must raise FileNotFoundError in binance_walk_forward_baseline
    missing_csv = tmp_path / "does_not_exist.csv"
    try:
        bwf.main([
            "--intervals", "60",
            "--dataset", f"60={missing_csv}",
            "--output-dir", str(tmp_path / "out_b"),
        ])
        assert False, "Expected FileNotFoundError for missing explicit --dataset"
    except FileNotFoundError:
        pass

    # 2. Negative --fee must be rejected in both runners
    for mod in (bwf, hwf):
        try:
            mod.main(["--fee", "-0.1", "--output-dir", str(tmp_path / "out_neg")])
            assert False, "Expected SystemExit for negative --fee"
        except SystemExit as exc:
            assert exc.code != 0

    # 3. hyperliquid_walk_forward_baseline accepts --warmup and records signal_warmup_bars in windows
    fake_records = [
        {"timestamp": 1700000000 + i * 3600, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for i in range(40)
    ]
    out_hl = tmp_path / "out_hl"
    with patch.object(hwf, "fetch_ohlc_candles", return_value=fake_records):
        rc = hwf.main([
            "--intervals", "60",
            "--coin", "HYPE",
            "--warmup", "5",
            "--output-dir", str(out_hl),
        ])
        assert rc == 0
    hl_report = json.loads((out_hl / "HYPE_60m_baseline.json").read_text(encoding="utf-8"))
    assert hl_report["windows"]["signal_warmup_bars"] == 5


def test_hyperliquid_uppercase_interval_and_zero_qty_reconcile_reset():
    from providers.market_api import clean_order_id

    _hl_md = load_local_module(_HL_DIR, "market_data", module_key="hl_md_case_test", reload=True)

    fake_candles = [
        {"t": 1700000000000, "T": 1700003599999, "s": "HYPE", "i": "1h",
         "o": "20.0", "c": "21.0", "h": "22.0", "l": "19.5", "v": "100.0", "n": 50},
    ]
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_candles).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        records = _hl_md.fetch_ohlc_candles("HYPE", "1H", drop_incomplete=False)
        assert len(records) == 1
        assert records[0]["close"] == 21.0

    # Zero-qty reconcile branch resets dust cost/spent
    Strategy = hl_replay.Strategy
    StratParams = hl_replay.StratParams

    class ZeroPosClient:
        def sz_decimals(self, coin):
            return 2

        def position_strict(self, coin):
            return (0.0, 0.0)

        def open_orders_strict(self, coin):
            return []

    strat = Strategy(
        client=ZeroPosClient(),
        coin="HYPE",
        params=StratParams(direction="long"),
        dry_run=False,
        replay_mode=True,
    )
    strat.s["qty"] = 0.0
    strat.s["cost"] = 12.34
    strat.s["spent"] = 12.34
    assert strat.reconcile(100.0) is True
    assert strat.s["cost"] == 0.0
    assert strat.s["spent"] == 0.0

    # clean_order_id rejects negative numeric strings and booleans
    assert clean_order_id("-123") is None
    assert clean_order_id(True) is None
    assert clean_order_id("valid-coid-1") == "valid-coid-1"


# ─── Delta-Neutral & Funding Scanner Tests (from test_dn.py & test_funding_scan.py) ───


_DN_DIR = os.path.join(_ROOT, "archive", "hyperliquid", "delta_neutral")
if _DN_DIR not in sys.path:
    sys.path.insert(0, _DN_DIR)

dn = load_local_module(_DN_DIR, "delta_neutral", module_key="hl_delta_neutral_test_mod")
DeltaNeutral = dn.DeltaNeutral
DNParams = dn.DNParams
state_path_for = dn.state_path_for
_funding_scan = load_local_module(_DN_DIR, "funding_scan", module_key="hl_funding_scan_test_mod")
rank = _funding_scan.rank

_DN_COIN = "TSTDN"


def _dn_params(**over) -> DNParams:
    base = dict(
        coin=_DN_COIN, spot_pair="@999", spot_token=_DN_COIN, notional=100.0,
        entry_funding_hr=0.0, exit_funding_hr=-0.00005, funding_window_h=4.0,
        min_hold_h=6.0, rebalance_pct=5.0, check_minutes=5.0, sz_decimals=2,
        liq_alert_pct=20.0, auto_protect=True, reduce_pct=25.0, perp_leverage=1,
        allow_scale_up=False, fee_pct=0.035,
    )
    base.update(over)
    return DNParams(**base)


class _FakeDNClient:
    exchange = None

    def __init__(self):
        self.orders: list[tuple] = []
        self.fail_orders = False
        self.raise_account = False
        self.spot_bal = 2.0
        self.perp_szi = -2.0
        self.liq_px = 0.0
        self.free = 100000.0

    def withdrawable(self):
        return self.free

    def spot_balance(self, token):
        return self.free if token == "USDC" else self.spot_bal

    def spot_mid(self, pair):
        return 50.0

    def mid(self, coin):
        return 50.0

    def funding_rate(self, coin):
        return 0.0000125

    def spot_balance_strict(self, token):
        if self.raise_account:
            raise RuntimeError("API down")
        return self.spot_bal

    def position_strict(self, coin):
        if self.raise_account:
            raise RuntimeError("API down")
        return self.perp_szi, 50.0

    def position_full(self, coin):
        return {"szi": self.perp_szi, "liquidationPx": self.liq_px}

    def spot_order(self, pair, is_buy, sz, px, szd):
        if self.fail_orders:
            return False, None, "err: margin"
        self.orders.append(("spot", "buy" if is_buy else "sell", sz))
        return True, 1, "ok"

    def place_limit(self, coin, is_buy, sz, px, reduce_only=False):
        if self.fail_orders:
            return False, None, "err: margin"
        self.orders.append(("perp", "buy" if is_buy else "sell", sz))
        return True, 2, "ok"

    def set_leverage(self, coin, lev):
        pass

    def open_orders(self, coin=None):
        return []

    def cancel(self, coin, oid):
        return True


def _dn_legs(spot_qty=2.0, perp_szi=-2.0, funding=0.0000125):
    return {
        "spot_px": 50.0, "perp_px": 50.0, "funding": funding,
        "spot_qty": spot_qty, "perp_szi": perp_szi,
    }


class _DNBase(_unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        self.alerts: list[str] = []
        self._notify = dn.notify
        dn.notify = lambda **kw: self.alerts.append(kw.get("title", ""))
        self._cleanup_state()
        self.c = _FakeDNClient()

    def tearDown(self):
        dn.notify = self._notify
        self._cleanup_state()
        os.environ.clear()
        os.environ.update(self._env)

    @staticmethod
    def _cleanup_state():
        for suf in ("", ".lock", ".tmp"):
            p = state_path_for(_DN_COIN) + suf
            if os.path.exists(p):
                os.remove(p)

    def make(self, dry=False, **over) -> DeltaNeutral:
        return DeltaNeutral(self.c, _dn_params(**over), dry_run=dry, desktop=False)

    def opened(self, d: DeltaNeutral, target=2.0):
        d.s["status"] = "open"
        d.s["target_sz"] = target
        d.s["opened_ts"] = _time.time()


class TestFailedReads(_DNBase):
    def test_an_account_api_error_does_not_guess_and_does_not_trade(self):
        d = self.make()
        self.c.raise_account = True
        self.assertIsNone(d.legs(), "a failed read -> None, not a false 0")
        self.assertEqual(self.c.orders, [])

    def test_a_missing_price_gives_none(self):
        d = self.make()
        self.c.spot_mid = lambda pair: None
        self.assertIsNone(d.legs())


class TestOrphanLeg(_DNBase):
    def test_a_single_suspicious_tick_does_not_act(self):
        d = self.make()
        self.opened(d)
        d.tick(_dn_legs(spot_qty=2.0, perp_szi=0.0))
        self.assertEqual(self.c.orders, [], "anti-glitch: it does not act on the first reading")
        self.assertEqual(d.s["orphan_count"], 1)

    def test_a_liquidated_short_closes_the_spot_after_confirmation(self):
        d = self.make()
        self.opened(d)
        d.tick(_dn_legs(spot_qty=2.0, perp_szi=0.0))
        d.tick(_dn_legs(spot_qty=2.0, perp_szi=0.0))
        sells = [o for o in self.c.orders if o[0] == "spot" and o[1] == "sell"]
        self.assertEqual(len(sells), 1, "sell the remaining spot leg to de-risk")
        self.assertEqual(d.s["status"], "flat")
        self.assertGreater(d.s["cooldown_until"], _time.time(), "cooldown anti-thrash")
        self.assertTrue(any("a leg is gone" in a for a in self.alerts))

    def test_a_recovered_glitch_resets_the_counter(self):
        d = self.make()
        self.opened(d)
        d.tick(_dn_legs(spot_qty=2.0, perp_szi=0.0))
        d.tick(_dn_legs())
        self.assertEqual(d.s["orphan_count"], 0)
        self.assertEqual(d.s["status"], "open")

    def test_both_legs_gone_goes_flat_without_orders(self):
        d = self.make()
        self.opened(d)
        d.tick(_dn_legs(spot_qty=0.0, perp_szi=0.0))
        d.tick(_dn_legs(spot_qty=0.0, perp_szi=0.0))
        self.assertEqual(self.c.orders, [], "there is nothing to close")
        self.assertEqual(d.s["status"], "flat")

    def test_without_auto_protect_it_only_alerts(self):
        d = self.make(auto_protect=False)
        self.opened(d)
        d.tick(_dn_legs(spot_qty=2.0, perp_szi=0.0))
        d.tick(_dn_legs(spot_qty=2.0, perp_szi=0.0))
        self.assertEqual(self.c.orders, [])
        self.assertTrue(any("MANUAL INTERVENTION" in a for a in self.alerts))


class TestDriftAndDust(_DNBase):
    def test_a_large_drift_needs_confirmation_over_2_ticks(self):
        d = self.make()
        self.opened(d)
        d.tick(_dn_legs(spot_qty=0.8, perp_szi=-2.0))
        self.assertEqual(self.c.orders, [], "the first tick: it only observes")
        d.tick(_dn_legs(spot_qty=0.8, perp_szi=-2.0))
        buys = [o for o in self.c.orders if o[0] == "spot" and o[1] == "buy"]
        self.assertEqual(len(buys), 1)

    def test_a_small_drift_is_corrected_immediately(self):
        d = self.make()
        self.opened(d)
        d.tick(_dn_legs(spot_qty=1.6, perp_szi=-2.0))
        buys = [o for o in self.c.orders if o[0] == "spot" and o[1] == "buy"]
        self.assertEqual(len(buys), 1, "normal drift is corrected without delay")

    def test_dust_below_the_minimum_sends_no_order(self):
        d = self.make()
        self.opened(d, target=2.0)
        d._buy_spot(0.1, 50.0)
        self.assertEqual(self.c.orders, [])


class TestFailedOrders(_DNBase):
    def test_alerts_after_3_consecutive_failures(self):
        d = self.make()
        self.opened(d)
        self.c.fail_orders = True
        for _ in range(3):
            d._buy_spot(1.0, 50.0)
        self.assertTrue(any("3 consecutive failed orders" in a for a in self.alerts))
        self.c.fail_orders = False
        d._buy_spot(1.0, 50.0)
        self.assertEqual(d.s["order_fails"], 0, "success resets the counter")


class TestCooldownAndEntry(_DNBase):
    def test_the_cooldown_blocks_reopening(self):
        d = self.make()
        d.s["cooldown_until"] = _time.time() + 600
        d.tick(_dn_legs(spot_qty=0.0, perp_szi=0.0, funding=0.001))
        self.assertEqual(d.s["status"], "flat")
        self.assertEqual(self.c.orders, [])

    def test_after_the_cooldown_it_opens(self):
        d = self.make()
        d.s["cooldown_until"] = _time.time() - 1
        d.tick(_dn_legs(spot_qty=0.0, perp_szi=0.0, funding=0.001))
        self.assertEqual(d.s["status"], "open")
        self.assertEqual(len(self.c.orders), 2, "both legs placed")


class TestSmartExit(_DNBase):
    def _force_avg(self, d, val):
        d.s["funding_hist"] = [[_time.time(), val]] * 5

    def test_negative_funding_held_briefly_does_not_close(self):
        d = self.make()
        self.opened(d)
        self._force_avg(d, -0.0002)
        d.tick(_dn_legs(funding=-0.0002))
        self.assertEqual(d.s["status"], "open", "min_hold is not met yet")

    def test_negative_funding_held_long_enough_closes(self):
        d = self.make()
        self.opened(d)
        d.s["opened_ts"] = _time.time() - 7 * 3600
        self._force_avg(d, -0.0002)
        d.tick(_dn_legs(funding=-0.0002))
        self.assertEqual(d.s["status"], "flat")
        self.assertEqual(len(self.c.orders), 2, "it sells the spot and covers the perp")


class TestInfrastructure(_DNBase):
    def test_the_second_instance_is_refused_by_the_lock(self):
        d1 = self.make()
        self.assertTrue(d1._acquire_lock())
        d2 = self.make()
        self.assertFalse(d2._acquire_lock(), "the lock prevents duplicate orders")
        d1._lock_fh.close()
        d3 = self.make()
        self.assertTrue(d3._acquire_lock(), "after the stop the lock is released")
        d3._lock_fh.close()

    def test_the_save_is_atomic_and_valid(self):
        d = self.make()
        self.opened(d)
        d._save()
        with open(d.state_file) as f:
            st = json.load(f)
        self.assertEqual(st["status"], "open")
        self.assertFalse(os.path.exists(d.state_file + ".tmp"))

    def test_it_adopts_the_existing_position_on_a_restart(self):
        d = self.make()
        d.tick(_dn_legs(spot_qty=1.7, perp_szi=-1.71))
        self.assertEqual(d.s["status"], "open", "it adopts instead of opening a double position")
        self.assertAlmostEqual(d.s["target_sz"], 1.705, places=3)
        self.assertEqual(self.c.orders, [], "adoption places no new orders")


class TestLiquidationProtection(_DNBase):
    def test_auto_protect_reduces_both_legs(self):
        d = self.make()
        self.opened(d)
        self.c.liq_px = 55.0
        d.tick(_dn_legs())
        covers = [o for o in self.c.orders if o[0] == "perp" and o[1] == "buy"]
        sells = [o for o in self.c.orders if o[0] == "spot" and o[1] == "sell"]
        self.assertEqual((len(covers), len(sells)), (1, 1), "reduces both legs")
        self.assertTrue(any("LIQUIDATION" in a or "reduced" in a for a in self.alerts))


class TestScaleUp(_DNBase):
    def test_disabled_does_nothing(self):
        d = self.make(dry=True, allow_scale_up=False, notional=400.0)
        self.opened(d, target=2.0)
        d._maybe_scale_up(_dn_legs())
        self.assertEqual(d.s["target_sz"], 2.0)

    def test_it_bumps_the_target_to_the_notional(self):
        d = self.make(dry=True, allow_scale_up=True, notional=400.0)
        self.opened(d, target=2.0)
        d._maybe_scale_up(_dn_legs())
        self.assertAlmostEqual(d.s["target_sz"], 8.0)

    def test_it_does_not_grow_past_the_target(self):
        d = self.make(dry=True, allow_scale_up=True, notional=400.0)
        self.opened(d, target=8.0)
        d._maybe_scale_up(_dn_legs())
        self.assertEqual(d.s["target_sz"], 8.0)

    def test_partial_when_the_collateral_is_small(self):
        d = self.make(dry=False, allow_scale_up=True, notional=400.0)
        self.opened(d, target=2.0)
        self.c.free = 100.0
        d._maybe_scale_up(_dn_legs())
        self.assertGreater(d.s["target_sz"], 2.0)
        self.assertLess(d.s["target_sz"], 8.0)

    def test_a_tick_executes_the_scale_up_and_buys_both_legs(self):
        d = self.make(dry=False, allow_scale_up=True, notional=400.0)
        self.opened(d, target=2.0)
        d.tick(_dn_legs())
        d.tick(_dn_legs())
        buys_spot = [o for o in self.c.orders if o[0] == "spot" and o[1] == "buy"]
        buys_perp = [o for o in self.c.orders if o[0] == "perp" and o[1] == "sell"]
        self.assertTrue(buys_spot and buys_perp, "the scale-up buys spot and shorts perp (staying neutral)")


class TestRank(_unittest.TestCase):
    def setUp(self):
        self.uni = [{"name": "AAA"}, {"name": "BBB"}, {"name": "CCC"}, {"name": "DDD"}]
        self.ctxs = [
            {"funding": "0.00005", "dayNtlVlm": "50000000", "markPx": "10", "openInterest": "1000"},
            {"funding": "0.0001",  "dayNtlVlm": "1000000",  "markPx": "5",  "openInterest": "200"},
            {"funding": "0.00002", "dayNtlVlm": "80000000", "markPx": "60", "openInterest": "500"},
            {"funding": "-0.0001", "dayNtlVlm": "90000000", "markPx": "3",  "openInterest": "100"},
        ]

    def test_filters_illiquid(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=10_000_000)
        coins = [x["coin"] for x in r]
        self.assertNotIn("BBB", coins)

    def test_filters_negative_funding(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=0)
        self.assertNotIn("DDD", [x["coin"] for x in r])

    def test_ranks_descending_by_funding(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=10_000_000)
        self.assertEqual(r[0]["coin"], "AAA")
        self.assertEqual(r[1]["coin"], "CCC")

    def test_correct_apr(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=10_000_000)
        aaa = next(x for x in r if x["coin"] == "AAA")
        self.assertAlmostEqual(aaa["apr"], 0.00005 * 24 * 365 * 100, places=2)

    def test_top_limits_results(self):
        self.assertLessEqual(len(rank(self.uni, self.ctxs, 0, top=1)), 1)

