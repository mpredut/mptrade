#!/usr/bin/env python3
"""
An investigation (23 Jul 2026): does it make sense to promote vol_1h_pct from shadow_signals
la o decizie REALA in tradeall.py, inlocuind pragurile FIXE de detectie a
of the price move (PRICE_CHANGE_THRESHOLD_EUR / PRICE_CHANGE_THRESHOLD_BIG_EUR,
used by check_price_change() for the SMALL/BIG windows) with ADAPTIVE thresholds
(K * vol_1h_pct, the same formula as shadow_signals.vol_1h_pct)?

Context (answering the user question "did we not do this because of the backtest?"):
There is NO earlier backtest that tests EXACTLY this idea.
offline/research/tradeall_trigger_gate/ (21-22 iul) a testat ALTE tipuri de schimbari —
relaxing the start/confirmation conditions of a trend that has already begun, a cooldown,
a quality signal on a 24h regression — and none beat the current variant
plus buy & hold over 329 days. But NONE of those variants replaced
the FIXED thresholds with ones scaled on REALISED volatility — this is a new
idea, analogous to the successful promotion of the adaptive re-entry threshold on Kraken
(offline/research/kraken_adaptive_thresholds/), merely applied to a different mechanism.

Methodology (the same rigour as offline/research/kraken_adaptive_thresholds/ and
Experiment 6 in tradeall_trigger_gate/):
  - Istoric REAL, 329 zile (cache_price_{symbol}.jsonl, ~7 min/tick).
  - check_price_change() itself is NOT modified — only the threshold value
    sent is replaced with K*vol_1h_pct, computed from EXACTLY the same window
    BIG window also used by shadow_signals.vol_1h_pct (real history, not separately simulated).
  - FALLBACK onto the real FIXED thresholds (from config, NOT arbitrary values) when
    warm-up (<20 points in the BIG window) — the same fail-safe pattern as in
    Kraken (STRAT_REENTRY_ADAPTIVE): an unavailable signal does not stop or alter
    the trading, it merely falls back to the fixed value.
  - K_BIG = K_SMALL * RATIO, unde RATIO = pragul BIG fix / pragul SMALL fix de
    today (~4.79) — we keep the ratio between the windows so as not to introduce a second
    an untested dimension in the sweep (the same principle as on Kraken: reentry and
    DCA tested SEPARATELY, each with a single multiplier).
  - Comparatie: PnL net (realizat + mark-to-market - comisioane) vs varianta
    the FIXED one (the default K) and against buy & hold, on BOTH symbols (BTC, TAO).

It reuses offline.backtests.tradeall.run_backtest() through the `threshold_provider` hook
(added 23 Jul as part of this investigation) — it no longer copies the
tick. Before this hook, this file had its own copy of the loop from
run_backtest() (a risk of silent drift from the "official" engine if that one
changes later — see offline/research/BACKTEST_CANDIDATES.md and the discussion in
session about unifying the backtests). The refactor was verified to
reproduce the old loop's results BIT FOR BIT, on real data, before
a inlocui vechea implementare.
"""
import os
import sys
import json
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.backtests import tradeall as tb
import shadow_signals

# Today's REAL FIXED values (from tradeall_config.env via tradeall SSOT) —
# used as (a) the warm-up fallback and (b) the "FIXED" baseline to compare against.
FIXED_SMALL = float(tb.ta.PRICE_CHANGE_THRESHOLD_EUR)
FIXED_BIG = float(tb.ta.PRICE_CHANGE_THRESHOLD_BIG_EUR)
RATIO = FIXED_BIG / FIXED_SMALL   # ~4.79


def run_adaptive(symbol, start_ts, end_ts, k_small, run_id):
    """A thin wrapper over offline.backtests.tradeall.run_backtest(): it builds a
    threshold_provider that returns K*vol_1h_pct (adaptive) instead of the fixed threshold,
    falling back to FIXED_SMALL/FIXED_BIG during the warm-up. k_small=None => a control
    run with the FIXED thresholds (equivalent to a normal run_backtest(), without the hook)."""
    k_big = None if k_small is None else k_small * RATIO
    warmup = {"n": 0}

    def _threshold_provider(window_small, window_big):
        if k_small is None:
            return FIXED_SMALL, FIXED_BIG
        vol1h = shadow_signals.vol_1h_pct(window_big.prices, window_big.sample_rate_sec)
        if vol1h is None:
            warmup["n"] += 1
            return FIXED_SMALL, FIXED_BIG
        return k_small * vol1h, k_big * vol1h

    tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "history",
                     quiet=True, kalman_primary=False, threshold_provider=_threshold_provider)

    pnl_path = os.path.join(ROOT, "logger", "backtest", run_id, "pnl.json")
    pnl = json.load(open(pnl_path, encoding="utf-8")) if os.path.exists(pnl_path) else {}
    pnl["warmup_ticks"] = warmup["n"]
    pnl["k_small"] = k_small
    pnl["k_big"] = round(k_big, 4) if k_big is not None else None
    with open(pnl_path, "w", encoding="utf-8") as pf:
        json.dump(pnl, pf, indent=1)
    sys.stderr.write(f"[{run_id}] P&L: {pnl}\n")
    return pnl


def _run_task(args):
    symbol, hist_start, k = args
    tag = "FIX" if k is None else f"k{k}"
    run_id = f"tradeall_adaptive_{symbol}_{tag}"
    t0 = time.time()
    pnl = run_adaptive(symbol, hist_start, None, k, run_id)
    sys.stderr.write(f"  ({time.time()-t0:.1f}s)\n")
    return (symbol, tag), pnl


if __name__ == "__main__":
    import concurrent.futures as _futures

    hist_start = datetime.strptime("2025-08-27", "%Y-%m-%d").timestamp()
    _k_env = os.environ.get("K_SWEEP", "0.1,0.2,0.3,0.5")
    K_SWEEP = [None] + [float(x) for x in _k_env.split(",") if x.strip()]   # None = control (FIX)

    for symbol in ("BTCUSDC", "TAOUSDC"):
        for _ in tb.load_ticks_history(symbol, hist_start, None):
            break

    tasks = [(symbol, hist_start, k) for symbol in ("BTCUSDC", "TAOUSDC") for k in K_SWEEP]
    t_all = time.time()
    max_workers = min(int(os.environ.get("WORKERS", 4)), os.cpu_count() or 2)
    with _futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = dict(ex.map(_run_task, tasks))

    sys.stderr.write(f"\n\n===== REZUMAT (wall total {time.time()-t_all:.1f}s) =====\n")
    for symbol in ("BTCUSDC", "TAOUSDC"):
        sys.stderr.write(f"\n--- {symbol} ---\n")
        for k in K_SWEEP:
            tag = "FIX" if k is None else f"k{k}"
            pnl = results[(symbol, tag)]
            sys.stderr.write(
                f"  {tag:6s}: net_total={pnl.get('net_total'):>10} buy_hold={pnl.get('buy_hold_net'):>10} "
                f"buys={pnl.get('buys')} sells={pnl.get('sells')} warmup_ticks={pnl.get('warmup_ticks')}\n")
