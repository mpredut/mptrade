#!/usr/bin/env python3
"""
Experiment 9 (isolated) -- does an RSI-oversold FILTER on DCA improve spot_dca?

Experiment 8 rejected RSI/Bollinger as a standalone TRIGGER. The one hypothesis left
from the assessment is the FILTER use: keep spot_dca's DCA logic exactly, but only allow
a DCA buy when RSI is oversold -- i.e. average down into genuine exhaustion, not every
-2% dip. This closes that door.

Isolated: does NOT modify spot_dca.py. It monkeypatches, in memory only:
  * strategies.spot_dca.Strategy.step        -> feed a Wilder-RSI tracker each bar close
  * strategies.spot_dca_rules.dca_price_hit  -> AND the existing DCA trigger with
    "RSI < threshold". Baseline leaves the tracker None, so both wrappers are no-ops and
    the run is byte-for-byte the unmodified strategy.

Data: the sparse 329-day history (cache_price_{symbol}.jsonl, record {"s","i":[ts_ms,px]})
resampled to HOURLY OHLC -- real intrabar H/L for fill modelling, and RSI(14) ~= 14h, a
sensible DCA timescale. Config mirrors the live HL profile (entry 350, dca 100, drop 2%,
max 7, TP 5%, budget 1050, stop 20). fee_pct=0.26 (the higher Kraken fee) deliberately
FAVORS the gate, which skips DCAs and thus fees -- if it cannot win there, it cannot win.

Metric: run_replay()["total"] (realized_net + open unrealized). The filter is worth it
only if a gated variant beats the ungated baseline on the full span AND both halves for a
symbol; otherwise its selectivity does not pay for the DCAs it skips.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import strategies.spot_engine as _strat
import strategies.spot_rules as sr
from kraken import replay as rp


class RsiGate:
    """Wilder RSI fed the bar closes; oversold() gates the DCA trigger."""

    def __init__(self, period, low):
        self.period, self.low = period, low
        self.prev = None
        self.ag = self.al = None
        self.seed = []
        self.rsi = None

    def update(self, price):
        if self.prev is None:
            self.prev = price
            return
        change = price - self.prev
        self.prev = price
        gain, loss = max(change, 0.0), max(-change, 0.0)
        if self.ag is None:
            self.seed.append((gain, loss))
            if len(self.seed) < self.period:
                return
            self.ag = sum(g for g, _ in self.seed) / self.period
            self.al = sum(l for _, l in self.seed) / self.period
        else:
            self.ag = (self.ag * (self.period - 1) + gain) / self.period
            self.al = (self.al * (self.period - 1) + loss) / self.period
        self.rsi = 100.0 if self.al == 0.0 else 100.0 - 100.0 / (1.0 + self.ag / self.al)

    def oversold(self):
        return self.rsi is not None and self.rsi < self.low


_gate = None   # RsiGate for a gated run, None for the baseline (wrappers become no-ops)

_orig_step = _strat.Strategy.step
def _step_feed_rsi(self, price, *args, **kwargs):
    if _gate is not None:
        _gate.update(float(price))
    return _orig_step(self, price, *args, **kwargs)

_orig_hit = sr.dca_price_hit
def _hit_gated(*a, **k):
    base = _orig_hit(*a, **k)
    if not base or _gate is None:
        return base
    return _gate.oversold()

_strat.Strategy.step = _step_feed_rsi
sr.dca_price_hit = _hit_gated


def load_ticks(symbol):
    path = os.path.join(ROOT, "cachedb", f"cache_price_sparse_{symbol}.jsonl")
    ticks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts_ms, price = rec["i"][0], rec["i"][1]
            ticks.append((float(ts_ms) / 1000.0, float(price)))
    ticks.sort()
    return ticks


def hourly_ohlc(ticks, start_ts, end_ts):
    buckets, order = {}, []
    for ts, p in ticks:
        if ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            break
        hour = int(ts // 3600)
        bar = buckets.get(hour)
        if bar is None:
            buckets[hour] = [p, p, p, p]
            order.append(hour)
        else:
            bar[1] = max(bar[1], p)
            bar[2] = min(bar[2], p)
            bar[3] = p
    return [tuple(buckets[h]) for h in order]


def make_params(**over):
    d = dict(currency="USDC", entry_amount=350.0, entry_discount_pct=0.2, dca_amount=100.0,
             dca_drop_pct=2.0, check_minutes=60.0, takeprofit_pct=5.0, max_budget=1050.0,
             max_dca_buys=7, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=20.0,
             adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=0.0, reentry_tolerance_pct=0.05,
             reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[])
    d.update(over)
    return _strat.StratParams(**d)


def run_one(bars, gate):
    global _gate
    _gate = gate
    try:
        res = rp.run_replay(bars, make_params(), fee_pct=0.26, bar_minutes=60.0,
                            initial_cash=1050.0)
    finally:
        _gate = None
    return res


CONFIGS = [("baseline", None), ("rsi30", (14, 30.0)), ("rsi40", (14, 40.0))]


if __name__ == "__main__":
    from datetime import datetime

    hist_start = datetime.strptime("2025-08-27", "%Y-%m-%d").timestamp()
    now = datetime.now().timestamp()
    mid = hist_start + (now - hist_start) / 2.0
    periods = [("full", hist_start, None), ("H1", hist_start, mid), ("H2", mid, now)]

    results = {}   # (symbol, period) -> {config: total}
    for symbol in ("BTCUSDC", "TAOUSDC", "ARBUSDC"):
        sys.stderr.write(f"\n=== {symbol} ===\n")
        ticks = load_ticks(symbol)
        for plabel, s, e in periods:
            bars = hourly_ohlc(ticks, s, e)
            if len(bars) < 50:
                sys.stderr.write(f"  {plabel}: only {len(bars)} bars, skipped\n")
                continue
            bh_pct = (bars[-1][3] / bars[0][3] - 1.0) * 100.0
            row = {}
            for name, gcfg in CONFIGS:
                gate = RsiGate(*gcfg) if gcfg else None
                res = run_one(bars, gate)
                row[name] = res
                sys.stderr.write(
                    f"  {symbol} {plabel:<4} {name:<8} total={res['total']!s:>10} "
                    f"net={res['net']!s:>10} fills={res['fills']:<4} "
                    f"open_qty={res['open_qty']}\n")
            sys.stderr.write(f"    ({plabel} bars={len(bars)} buy&hold={bh_pct:+.1f}%  "
                             f"delta rsi30-base={row['rsi30']['total'] - row['baseline']['total']:+.2f}  "
                             f"rsi40-base={row['rsi40']['total'] - row['baseline']['total']:+.2f})\n")
            results[(symbol, plabel)] = {k: v["total"] for k, v in row.items()}

    sys.stderr.write("\n\n===== VERDICT: does a gate beat baseline on full AND both halves? =====\n")
    any_pass = False
    for symbol in ("BTCUSDC", "TAOUSDC", "ARBUSDC"):
        for gate in ("rsi30", "rsi40"):
            deltas = {}
            for plabel in ("full", "H1", "H2"):
                r = results.get((symbol, plabel))
                deltas[plabel] = (r[gate] - r["baseline"]) if r else None
            vals = [deltas["full"], deltas["H1"], deltas["H2"]]
            consistent = all(v is not None and v > 0 for v in vals)
            any_pass = any_pass or consistent
            mark = "PASS" if consistent else "----"
            sys.stderr.write(f"  [{mark}] {symbol:<8} {gate:<6} "
                             f"delta full={vals[0]!s:>10} H1={vals[1]!s:>10} H2={vals[2]!s:>10}\n")
    sys.stderr.write("\nNo RSI-oversold DCA filter beats the ungated baseline consistently.\n"
                     if not any_pass else "\nA gate consistently beats baseline -- inspect further.\n")
