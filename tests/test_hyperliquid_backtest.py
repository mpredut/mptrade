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
