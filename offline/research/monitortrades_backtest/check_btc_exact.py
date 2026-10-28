#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import pricewindow as pw
from cachepaths import cache_path
from providers.replay_provider import load_price_series
import offline.research.monitortrades_backtest.run_replay_backtest as rrb

series = load_price_series(cache_path("cache_sparse_price_BTCUSDC.jsonl"), "BTCUSDC")
cfg = rrb.SYMBOLS["BTCUSDC"]

for pwr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for win_s in [90.0, 150.0, 222.0, 300.0]:
        os.environ["MPTRADE_STEP_WOLS_POWER"] = str(pwr)
        rrb._TREND_TAPE_CACHE.clear()
        tape = rrb.prewarm_trend_series("BTCUSDC", series, trend_window_seconds=win_s)
        key = rrb._series_cache_key("BTCUSDC", series, rrb.DEFAULT_WINDOW_SECONDS)
        rrb._TREND_TAPE_CACHE[key] = tape
        res = rrb.run_symbol("BTCUSDC", cfg["params"], cfg["base"], quiet=True, series=series)
        print(f"BTCUSDC exact p={pwr:.1f} win={win_s:>5.0f}s -> net={res['net']:>+8.2f} ({res['buys']}B/{res['sells']}S) up%={100*sum(tape)/len(tape):.2f}%")
os.environ.pop("MPTRADE_STEP_WOLS_POWER", None)
