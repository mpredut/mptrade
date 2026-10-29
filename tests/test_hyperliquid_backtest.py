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


def test_hyperliquid_market_data_fetchers():
    import json
    from unittest.mock import MagicMock, patch

    _spec = importlib.util.spec_from_file_location("hl_md_test", os.path.join(_HL_DIR, "market_data.py"))
    _hl_md = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_hl_md)
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
    import json
    from unittest.mock import patch
    import importlib.util
    _wf_spec = importlib.util.spec_from_file_location(
        "hl_wf_baseline", os.path.join(_ROOT, "offline", "runners", "hyperliquid_walk_forward_baseline.py")
    )
    _wf_mod = importlib.util.module_from_spec(_wf_spec)
    _wf_spec.loader.exec_module(_wf_mod)

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
    import json
    from unittest.mock import patch
    import importlib.util
    _wf_spec = importlib.util.spec_from_file_location(
        "hl_wf_baseline", os.path.join(_ROOT, "offline", "runners", "hyperliquid_walk_forward_baseline.py")
    )
    _wf_mod = importlib.util.module_from_spec(_wf_spec)
    _wf_spec.loader.exec_module(_wf_mod)

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
    from offline.backtests.execution import ExecutionModel
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
    spec_pa = importlib.util.spec_from_file_location("hl_pa_module", os.path.join(_HL_DIR, "price_analysis.py"))
    price_analysis = importlib.util.module_from_spec(spec_pa)
    spec_pa.loader.exec_module(price_analysis)

    spec_sig = importlib.util.spec_from_file_location("hl_sig_module", os.path.join(_HL_DIR, "signals.py"))
    signals = importlib.util.module_from_spec(spec_sig)
    spec_sig.loader.exec_module(signals)

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


def test_walk_forward_compare_preserves_zero_fee():
    from offline.runners.walk_forward_compare import _build_venue_evaluator

    _, krk_fee, _ = _build_venue_evaluator("kraken", {"fee_pct_per_leg": 0.0, "strategy_params": {}})
    assert krk_fee == 0.0

    _, hl_fee, _ = _build_venue_evaluator("hyperliquid", {"fee_pct_per_leg": 0.0, "strategy_params": {"fee_pct": 0.035}})
    assert hl_fee == 0.0


def test_hyperliquid_sz_decimals_spot_fallback_and_exact_sl_boundary():
    import types
    hl_client_spec = importlib.util.spec_from_file_location("hl_client_test_mod", os.path.join(_HL_DIR, "hl_client.py"))
    hl_client_mod = importlib.util.module_from_spec(hl_client_spec)
    hl_client_spec.loader.exec_module(hl_client_mod)

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
    import json
    from unittest.mock import patch
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




