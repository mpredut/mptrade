#!/usr/bin/env python3
"""
Experiment 8 (isolated, it does NOT modify tradeall.py) -- test whether an RSI /
Bollinger %B MEAN-REVERSION signal is worth adding, at the user's request.

Rationale: every trend/threshold change tried so far (Experiments 1-7) failed to beat
buy & hold across 12h-329 days -- but all of them were momentum/trend variants. RSI and
Bollinger %B are a QUALITATIVELY different, mean-reversion (contrarian) signal that has
not been tried: buy exhaustion at oversold / the lower band, sell into overbought / the
upper band. Neither indicator exists in the live code; this measures whether the idea
has any edge before any of it is wired anywhere.

Method (matches Experiment 6):
  * SPARSE history `cache_price_{symbol}.jsonl` (~7 min/tick, from 2025-08-27). RSI(14)
    spans ~14*7min ~= 1.6h and %B(20) ~= 2.3h -- slow enough for 7-min sampling, unlike
    tradeall's native 2.7-min constants.
  * Replay via offline/backtests/tradeall.py (simulated execution, no network).
  * Fire ONCE per signal regime, with MIN_RETRY_INTERVAL_SEC between BLOCKED retries
    (the retry-storm defect Experiment 5 exposed).
  * OVERFIT CHECK: each variant is run on the FULL span AND on the first / second half
    separately. An edge that only appears on one half is small-sample luck, the exact
    trap that sank the optimistic 7-day result in Experiment 5.

Benchmark: pnl["net_total"] (strategy) vs pnl["buy_hold_net"] (buy & hold on the same
standard quantity over the same interval). To be worth pursuing, a variant must beat
buy & hold on the full span AND on both halves.
"""
import os
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.backtests import tradeall as tb
import tradeall as ta

MIN_RETRY_INTERVAL_SEC = 1800.0   # 30 min between BLOCKED attempts, as in Experiment 6

STATS = {}


def _rsi_seed_and_step():
    """Wilder's RSI as a closure: feed prices in order, returns RSI once seeded."""


class MeanReversionTracker:
    """Compute RSI (Wilder) and Bollinger %B on the sparse price stream and derive a
    contrarian sign: +1 oversold (buy), -1 overbought (sell), 0 neutral."""

    def __init__(self, tag, symbol, *, mode, rsi_period=14, rsi_low=30.0, rsi_high=70.0,
                 bb_period=20, bb_k=2.0, pb_low=0.0, pb_high=1.0):
        self.mode = mode                      # "rsi" | "pb" | "both"
        self.rsi_period = rsi_period
        self.rsi_low, self.rsi_high = rsi_low, rsi_high
        self.bb_period = bb_period
        self.bb_k = bb_k
        self.pb_low, self.pb_high = pb_low, pb_high

        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self._seed_gains = []
        self.bb_buf = deque(maxlen=bb_period)

        self.sign = 0
        self.fired_up = False
        self.fired_down = False
        self.last_attempt_up_ts = None
        self.last_attempt_down_ts = None
        self.stats = STATS[tag]

    # -- indicators -----------------------------------------------------------
    def _rsi(self, price):
        if self.prev_price is None:
            self.prev_price = price
            return None
        change = price - self.prev_price
        self.prev_price = price
        gain, loss = max(change, 0.0), max(-change, 0.0)
        if self.avg_gain is None:                     # still seeding the first period
            self._seed_gains.append((gain, loss))
            if len(self._seed_gains) < self.rsi_period:
                return None
            self.avg_gain = sum(g for g, _ in self._seed_gains) / self.rsi_period
            self.avg_loss = sum(l for _, l in self._seed_gains) / self.rsi_period
        else:                                         # Wilder smoothing
            self.avg_gain = (self.avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self.avg_loss = (self.avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
        if self.avg_loss == 0.0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _percent_b(self, price):
        self.bb_buf.append(price)
        if len(self.bb_buf) < self.bb_period:
            return None
        mean = sum(self.bb_buf) / len(self.bb_buf)
        var = sum((p - mean) ** 2 for p in self.bb_buf) / len(self.bb_buf)
        sd = var ** 0.5
        if sd == 0.0:
            return 0.5
        upper, lower = mean + self.bb_k * sd, mean - self.bb_k * sd
        return (price - lower) / (upper - lower)

    # -- combined contrarian sign --------------------------------------------
    def update_sign(self, price):
        oversold = overbought = None
        if self.mode == "rsi":
            rsi = self._rsi(price)
            if rsi is None:
                return self.sign
            oversold, overbought = rsi < self.rsi_low, rsi > self.rsi_high
        elif self.mode == "pb":
            pb = self._percent_b(price)
            if pb is None:
                return self.sign
            oversold, overbought = pb < self.pb_low, pb > self.pb_high
        else:                                          # "both" -> agreement required
            rsi = self._rsi(price)
            pb = self._percent_b(price)
            if rsi is None or pb is None:
                return self.sign
            oversold = rsi < self.rsi_low and pb < self.pb_low
            overbought = rsi > self.rsi_high and pb > self.pb_high
        new_sign = 1 if oversold else (-1 if overbought else 0)
        if new_sign != self.sign:
            self.sign = new_sign
            self.fired_up = False
            self.fired_down = False
            self.stats["sign_changes"] += 1
        return self.sign


def _make_stats(tag):
    return STATS.setdefault(tag, {
        "sign_changes": 0,
        "fire_attempts_up": 0, "fire_confirmed_up": 0, "fire_blocked_up": 0, "fire_skipped_cooldown_up": 0,
        "fire_attempts_down": 0, "fire_confirmed_down": 0, "fire_blocked_down": 0, "fire_skipped_cooldown_down": 0,
    })


_trackers = {}


def make_logic(tag):
    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price):
        tr = _trackers.get((tag, symbol))
        ts = trend_state._now()
        sign = tr.update_sign(current_price)
        stats = tr.stats

        if sign > 0 and not tr.fired_up:              # oversold -> contrarian BUY
            can_retry = (tr.last_attempt_up_ts is None
                         or (ts - tr.last_attempt_up_ts) >= MIN_RETRY_INTERVAL_SEC)
            if not can_retry:
                stats["fire_skipped_cooldown_up"] += 1
            else:
                tr.last_attempt_up_ts = ts
                stats["fire_attempts_up"] += 1
                if enable:
                    result = ta._fire_order(symbol, "BUY", current_price, f"{tag}_oversold",
                                             safeback_seconds=14 * 24 * 3600 + 60, force=False,
                                             cancelorders=True, hours=1)
                    if result is not None:
                        tr.fired_up, tr.fired_down = True, False
                        stats["fire_confirmed_up"] += 1
                    else:
                        stats["fire_blocked_up"] += 1

        if sign < 0 and not tr.fired_down:            # overbought -> contrarian SELL
            can_retry = (tr.last_attempt_down_ts is None
                         or (ts - tr.last_attempt_down_ts) >= MIN_RETRY_INTERVAL_SEC)
            if not can_retry:
                stats["fire_skipped_cooldown_down"] += 1
            else:
                tr.last_attempt_down_ts = ts
                stats["fire_attempts_down"] += 1
                if enable:
                    result = ta._fire_order(symbol, "SELL", current_price, f"{tag}_overbought",
                                             safeback_seconds=14 * 24 * 3600 + 60, force=False,
                                             cancelorders=True, hours=1)
                    if result is not None:
                        tr.fired_down, tr.fired_up = True, False
                        stats["fire_confirmed_down"] += 1
                    else:
                        stats["fire_blocked_down"] += 1

    return logic_variant


VARIANTS = {
    # tag -> tracker kwargs
    "RSI14_30_70":   dict(mode="rsi", rsi_period=14, rsi_low=30.0, rsi_high=70.0),
    "RSI14_20_80":   dict(mode="rsi", rsi_period=14, rsi_low=20.0, rsi_high=80.0),
    "PB20_k2":       dict(mode="pb", bb_period=20, bb_k=2.0, pb_low=0.0, pb_high=1.0),
    "RSI_AND_PB":    dict(mode="both", rsi_period=14, rsi_low=35.0, rsi_high=65.0,
                          bb_period=20, bb_k=2.0, pb_low=0.15, pb_high=0.85),
}


def run_variant(tag, symbol, start_ts, end_ts, period_label):
    run_tag = f"{tag}__{period_label}"
    _make_stats(run_tag)
    _trackers[(run_tag, symbol)] = MeanReversionTracker(run_tag, symbol, **VARIANTS[tag])
    orig_logic = ta.logic
    orig_logic_small = getattr(ta, "logic_small", None)
    ta.logic = make_logic(run_tag)
    tb.ta.logic = ta.logic
    ta.logic_small = lambda *a, **kw: None
    tb.ta.logic_small = ta.logic_small

    run_id = f"experiment8_{tag}_{symbol}_{period_label}"
    import shutil
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    try:
        tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "history",
                         quiet=True, kalman_primary=False)
    finally:
        ta.logic = orig_logic
        tb.ta.logic = orig_logic
        if orig_logic_small is not None:
            ta.logic_small = orig_logic_small
            tb.ta.logic_small = orig_logic_small
    elapsed = time.time() - t0

    import json
    pnl_path = os.path.join(out_dir, "pnl.json")
    pnl = json.load(open(pnl_path)) if os.path.exists(pnl_path) else {}
    net = pnl.get("net_total")
    bh = pnl.get("buy_hold_net")
    edge = (net - bh) if (net is not None and bh is not None) else None
    st = STATS[run_tag]
    fires = st["fire_confirmed_up"] + st["fire_confirmed_down"]
    sys.stderr.write(
        f"  {tag:<12} {symbol:<8} {period_label:<5} "
        f"net={net!s:>10} bh={bh!s:>10} edge={edge!s:>10} "
        f"fires={fires:<4} (buy {st['fire_confirmed_up']}/sell {st['fire_confirmed_down']}, "
        f"blocked {st['fire_blocked_up'] + st['fire_blocked_down']})\n")
    return {"tag": tag, "symbol": symbol, "period": period_label,
            "net_total": net, "buy_hold_net": bh, "edge": edge, "fires": fires}


def _run_task(args):
    return run_variant(*args)


if __name__ == "__main__":
    import concurrent.futures as _futures
    from datetime import datetime

    hist_start = datetime.strptime("2025-08-27", "%Y-%m-%d").timestamp()
    now = time.time()
    mid = hist_start + (now - hist_start) / 2.0
    periods = [("full", hist_start, None), ("H1", hist_start, mid), ("H2", mid, now)]

    for symbol in ("BTCUSDC", "TAOUSDC"):
        for _label, s, e in periods:
            for _ in tb.load_ticks_history(symbol, s, e):
                break

    tasks = [
        (tag, symbol, s, e, label)
        for symbol in ("BTCUSDC", "TAOUSDC")
        for tag in VARIANTS
        for label, s, e in periods
    ]
    max_workers = min(int(os.environ.get("WORKERS", 4)), os.cpu_count() or 2)
    with _futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_run_task, tasks))

    # Verdict: a variant is worth pursuing only if it beats buy & hold on the FULL span
    # AND on BOTH halves for a symbol (consistency, not a single lucky regime).
    sys.stderr.write("\n\n===== VERDICT (positive edge on full AND both halves) =====\n")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["tag"], r["symbol"]), {})[r["period"]] = r["edge"]
    any_pass = False
    for (tag, symbol), e in sorted(by_key.items()):
        vals = [e.get("full"), e.get("H1"), e.get("H2")]
        consistent = all(v is not None and v > 0 for v in vals)
        mark = "PASS" if consistent else "----"
        if consistent:
            any_pass = True
        sys.stderr.write(f"  [{mark}] {tag:<12} {symbol:<8} "
                         f"edge full={vals[0]!s:>10} H1={vals[1]!s:>10} H2={vals[2]!s:>10}\n")
    sys.stderr.write("\nNo variant beats buy & hold consistently.\n" if not any_pass
                     else "\nAt least one variant is worth a closer look.\n")
