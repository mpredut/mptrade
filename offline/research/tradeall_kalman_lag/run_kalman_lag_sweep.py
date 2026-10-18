#!/usr/bin/env python3
"""
It investigates whether KALMAN_SAMPLE_SEC=60 (the sampling rate of the Kalman filter
from shadow_signals.py, which feeds the PRIMARY KALMAN plus the gate in tradeall.py)
introduces too much delay for fast price moves (23 Jul 2026).

Context: analiza unui episod REAL (BTC, 23 iul, 10:10:13-10:12:47) a aratat ca
a fall of -0.53% happened almost entirely BEFORE Kalman confirmed
"DOWN" (~90% of the move already consumed by the time it transitioned), and "DOWN"
lasted only 62s before returning to "FLAT" right at the local low -- exactly
before a rebound in the price (order_outcomes_2026-07-23.log confirms a
incercare reala de SELL "kalman_primary_down" la acel moment, refuzata
"no_fill" -- no real money moved this time, but the mechanism is live).

The structural cause: at 60s/sample, a move that plays out in ~90-150s
(the case above) cannot be seen sooner than ~1-2 periods of
esantionare, indiferent de pragurile de incredere (CONF_ENTER/CONF_EXIT).
KALMAN_SAMPLE_SEC=60 was chosen explicitly on 17 Jul in order to reduce the noise
(la 1s: 2868 tranzitii/zi pe BTC -- prea multe, palpaituri). Reducerea lui
risks reintroducing that noise -- this script quantifies exactly that
trade-off (latency versus noise) on REAL data, before any change.

It reuses the kalman_primary mode that ALREADY EXISTS and is validated in
offline/backtests/tradeall.py (Kalman drives BUY/SELL directly on transitions, through
broker.place_order_smart/sell_all) -- no new loop is built,
it only monkeypatches shadow_signals.KALMAN_SAMPLE_SEC before each
run (the functions read the global on every call, so the change takes effect
immediately, with no need to reload the module).

Methodology: a REAL 329-day history (cache_price_{symbol}.jsonl), the same
archive as offline/research/tradeall_trigger_gate/Experiment 6 and
offline/research/tradeall_adaptive_thresholds/. Sweep pe KALMAN_SAMPLE_SEC, comparat
pe PnL net (kalman_primary) + numarul de tranzitii Kalman logate (proxy
directly for noise/flapping) plus buy & hold over the same interval.

It does NOT modify tradeall.py, offline/backtests/tradeall.py or shadow_signals.py on disk --
only an in-memory monkeypatch, for the lifetime of the script. It never runs
against the real network.
"""
import os
import sys
import json
import shutil
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.backtests import tradeall as tb
import shadow_signals

# 60.0 = today's LIVE value (the control). The rest: faster (20) and slower
# (90/150), so we can see which way the latency<->noise trade-off moves.
# Reduced to 4 values (from 5) after the sanity check showed ~40 min/run
# on the full history (888k ticks/symbol) -- 4x2=8 runs, not 10.
SAMPLE_SEC_SWEEP = [20.0, 60.0, 90.0, 150.0]


def run_variant(symbol, start_ts, end_ts, sample_sec, run_id):
    orig_sample_sec = shadow_signals.KALMAN_SAMPLE_SEC
    shadow_signals.KALMAN_SAMPLE_SEC = sample_sec
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    try:
        tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "history",
                         quiet=True, kalman_primary=True)
    finally:
        shadow_signals.KALMAN_SAMPLE_SEC = orig_sample_sec
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    pnl_path = os.path.join(out_dir, "pnl.json")
    pnl = {}
    if os.path.exists(pnl_path):
        with open(pnl_path, encoding="utf-8") as fh:
            pnl = json.load(fh)

    # Count logged Kalman transitions (direct proxy for noise/instability).
    shadow_log = os.path.join(out_dir, "tradeall_shadow.log")
    n_transitions = 0
    if os.path.exists(shadow_log):
        with open(shadow_log, encoding="utf-8") as f:
            n_transitions = sum(1 for _ in f)

    pnl["sample_sec"] = sample_sec
    pnl["n_kalman_transitions"] = n_transitions
    with open(pnl_path, "w", encoding="utf-8") as pf:
        json.dump(pnl, pf, indent=1)
    sys.stderr.write(f"[{run_id}] ({elapsed:.1f}s) PnL: {pnl}\n")
    return pnl


def _run_task(args):
    symbol, hist_start, sec = args
    run_id = f"tradeall_kalman_lag_{symbol}_s{int(sec)}"
    pnl = run_variant(symbol, hist_start, None, sec, run_id)
    return (symbol, sec), pnl


if __name__ == "__main__":
    import concurrent.futures as _futures

    hist_start = datetime.strptime("2025-08-27", "%Y-%m-%d").timestamp()
    for symbol in ("BTCUSDC", "TAOUSDC"):
        for _ in tb.load_ticks_history(symbol, hist_start, None):
            break

    tasks = [(symbol, hist_start, sec) for symbol in ("BTCUSDC", "TAOUSDC") for sec in SAMPLE_SEC_SWEEP]
    t_all = time.time()
    max_workers = min(int(os.environ.get("WORKERS", 4)), os.cpu_count() or 2)
    with _futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = dict(ex.map(_run_task, tasks))

    sys.stderr.write(f"\n\n===== REZUMAT (wall total {time.time()-t_all:.1f}s) =====\n")
    for symbol in ("BTCUSDC", "TAOUSDC"):
        sys.stderr.write(f"\n--- {symbol} ---\n")
        for sec in SAMPLE_SEC_SWEEP:
            pnl = results[(symbol, sec)]
            sys.stderr.write(
                f"  sample_sec={sec:>5}: net_total={pnl.get('net_total')} buy_hold={pnl.get('buy_hold_net')} "
                f"buys={pnl.get('buys')} sells={pnl.get('sells')} transitions={pnl.get('n_kalman_transitions')}\n")
