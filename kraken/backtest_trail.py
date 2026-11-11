#!/usr/bin/env python3
"""
backtest_trail.py -- compare fixed TP with trend-following trailing TP on Kraken OHLC.

The question is whether a fixed +5% TP sells too early and misses trends, while a
trailing TP (once profitable, sell only after price falls X% from its peak) captures
more. The comparison uses multiple pairs rather than only HYPE, plus sequential
walk-forward segments. TOTAL is realized P&L plus the open position, versus buy-and-hold.

  python3 backtest_trail.py                     # comparison table across several pairs
  python3 backtest_trail.py --wf HYPEUSD        # walk-forward on a single pair
  python3 backtest_trail.py --intraday HYPEUSD  # pattern by hour of day
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import normalize_epoch_sec, safe_float, safe_int, safe_nonneg_int, safe_pos_float
from kraken.backtest import _bt_params, fetch_candles, simulate  # noqa: F401
from kraken.market_data import fetch_ohlc_candles
from strategies import spot_replay as rp
from offline.backtests.metrics import buy_hold_pct as bh


def _resolve_reentry_drop(P: dict) -> float:
    re_drop = P.get("reentry_drop")
    if re_drop is None:
        re_drop = P.get("reentry")
    if re_drop is None:
        return 0.0
    val = safe_float(re_drop, default=0.0)
    return max(0.0, val) if val is not None else 0.0


def fetch_with_time(pair, interval):
    """Fetch completed candles with timestamps for intraday distribution analysis."""
    records = fetch_ohlc_candles(pair, interval, drop_incomplete=True)
    return [(r["timestamp"], r["open"], r["high"], r["low"], r["close"]) for r in records]


def simulate_trail(ohlc, P, *, warmup_ohlc=()):
    """Match ``backtest.simulate`` except for trailing exit behavior.

    Delegates to the faithful replay engine (rp.run_replay) which runs
    strategies.spot_engine.Strategy directly with trailing take-profit enabled,
    eliminating parity bugs and duplicate simulation drift.
    """
    trail_val = P.get("trail") if P.get("trail") is not None else 3.0
    over = {
        "tp_trend_hold": True,
        "tp_trail_adaptive": False,
        "tp_trail_pct": trail_val,
        "tp_mdhtp_enabled": False,
        "tp_dynamic_flat": False,
        "reentry_drop_pct": _resolve_reentry_drop(P),
    }
    params = _bt_params(P, over)
    return rp.run_replay(
        ohlc,
        params,
        fee_pct=P.get("fee") if P.get("fee") is not None else 0.25,
        bar_minutes=P.get("interval") if P.get("interval") is not None else 60,
        warmup_ohlc=warmup_ohlc,
        intrabar_high=True,
    )


def simulate_fixed_tp(ohlc, P, *, warmup_ohlc=()):
    """Run fixed-TP replay with trailing, MDHTP, and dynamic flat TP disabled."""
    over = {
        "tp_trend_hold": False,
        "tp_mdhtp_enabled": False,
        "tp_dynamic_flat": False,
        "reentry_drop_pct": _resolve_reentry_drop(P),
    }
    params = _bt_params(P, over)
    return rp.run_replay(
        ohlc,
        params,
        fee_pct=P.get("fee") if P.get("fee") is not None else 0.25,
        bar_minutes=P.get("interval") if P.get("interval") is not None else 60,
        warmup_ohlc=warmup_ohlc,
    )


def base_params(budget=3000):
    return dict(entry=300, dca=150, disc=0.8, drop=1.0, tp=5.0, maxdca=10,
                budget=budget, fee=0.25, sl=7.0, trail=3.0)


def _eval_fixed_vs_trail(ohlc, params, *, warmup_ohlc=()):
    budget = params["budget"]
    h = bh(ohlc)
    f = simulate_fixed_tp(ohlc, params, warmup_ohlc=warmup_ohlc)["total"] / budget * 100
    t = simulate_trail(ohlc, params, warmup_ohlc=warmup_ohlc)["total"] / budget * 100
    win = "tie" if abs(t - f) <= 1e-9 else ("trailing" if t > f else "TP-fix")
    return h, f, t, win


def run_compare(pairs, intervals):
    P = base_params()
    print(f"params: entry={P['entry']} dca={P['dca']} drop={P['drop']}% sl={P['sl']}% | "
          f"FIX tp={P['tp']}% | TRAIL act={P['tp']}% trail={P['trail']}%  (fee {P['fee']}%/leg)")
    print(f"{'pair':<9} {'intv':<5} {'bars':<5} {'buy&hold':>9} {'TP-fix':>8} {'trailing':>9}  winner")
    fixw = trailw = ties = 0
    for pair in pairs:
        for iv in intervals:
            try:
                ohlc = fetch_candles(pair, iv)
            except Exception as e:  # noqa: BLE001
                print(f"{pair:<9} {iv:<5} ERROR: {e}"); continue
            if len(ohlc) < 30:
                print(f"{pair:<9} {iv:<5} not enough data ({len(ohlc)})"); continue
            h, f, t, win = _eval_fixed_vs_trail(ohlc, {**P, "interval": iv})
            if win == "tie":
                ties += 1
            elif win == "trailing":
                trailw += 1
            else:
                fixw += 1
            print(f"{pair:<9} {iv:<5} {len(ohlc):<5} {h:>+8.1f}% {f:>+7.2f}% {t:>+8.2f}%  {win}")
    print(f"\nTRAILING beats TP-fix in {trailw}/{trailw+fixw+ties} cases ({ties} ties)")


def run_wf(pair, interval, segments=4):
    seg_val = safe_int(segments, default=4)
    segments = max(1, seg_val if seg_val is not None else 4)
    P = {**base_params(), "interval": interval}
    try:
        ohlc = fetch_candles(pair, interval)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}"); return
    n = len(ohlc); seg = max(1, n // segments)
    print(f"=== WALK-FORWARD {pair} interval={interval}m ({n} bars, {segments} segments) ===")
    print(f"{'segment':<9} {'buy&hold':>9} {'TP-fix':>8} {'trailing':>9}  winner")
    fixw = trailw = ties = 0
    for i in range(segments):
        start_idx = i * seg
        end_idx = (i + 1) * seg if i < segments - 1 else n
        s = ohlc[start_idx:end_idx]
        if len(s) < 20:
            continue
        warmup = ohlc[max(0, start_idx - 40):start_idx]
        h, f, t, win = _eval_fixed_vs_trail(s, P, warmup_ohlc=warmup)
        if win == "tie":
            ties += 1
        elif win == "trailing":
            trailw += 1
        else:
            fixw += 1
        print(f"#{i+1:<8} {h:>+8.1f}% {f:>+7.2f}% {t:>+8.2f}%  {win}")
    print(f"\nTRAILING beats TP-fix in {trailw}/{trailw+fixw+ties} segments ({ties} ties, sequential out-of-sample)")


def run_intraday(pair, tz_offset=3):
    """Report average (close-open)/open return by local hour.

    The sample is small (~30 days at 1h, or ~30 observations per hour), so the
    result should be treated skeptically.
    """
    tz_val = safe_int(tz_offset, default=3)
    tz_offset = tz_val if tz_val is not None else 3
    try:
        rows = fetch_with_time(pair, 60)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}"); return
    by_hour = collections.defaultdict(list)
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        ts, o, _h, _l, c = row[:5]
        ts_sec = normalize_epoch_sec(ts, default=None)
        o_f = safe_pos_float(o, default=None)
        c_f = safe_float(c, default=None)
        if ts_sec is None or o_f is None or c_f is None:
            continue
        ts_i = safe_int(ts_sec, default=0)
        hr = (ts_i // 3600 + tz_offset) % 24
        by_hour[hr].append((c_f - o_f) / o_f * 100)
    print(f"=== INTRADAY {pair} (LOCAL hour UTC+{tz_offset}, {len(rows)} hourly bars) ===")
    print(f"{'hour':<5} {'n':<5} {'avg.return/hour':>14} {'% positive bars':>16}")
    for hr in range(24):
        v = by_hour.get(hr, [])
        if not v:
            continue
        avg = statistics.fmean(v); pos = 100*sum(1 for x in v if x > 0)/len(v)
        bar = "#" * safe_nonneg_int(abs(avg) * 20, default=0)
        sign = "+" if avg >= 0 else "-"
        print(f"{hr:>02}:00 {len(v):<5} {avg:>+13.3f}% {pos:>14.0f}%  {sign}{bar}")
    # Aggregate the 06:00-12:00 morning window (hours 06..11).
    morn = [x for hr in range(6, 12) for x in by_hour.get(hr, [])]
    rest = [x for hr in list(range(0, 6)) + list(range(12, 24)) for x in by_hour.get(hr, [])]
    if morn and rest:
        print(f"\n  morning 06-12  : avg {statistics.fmean(morn):+.3f}%/hour ({len(morn)} obs)")
        print(f"  rest of day    : avg {statistics.fmean(rest):+.3f}%/hour ({len(rest)} obs)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", metavar="PAIR", help="walk-forward on a single pair")
    ap.add_argument("--intraday", metavar="PAIR", help="the pattern by hour of day")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    if args.wf:
        run_wf(args.wf, args.interval)
    elif args.intraday:
        run_intraday(args.intraday)
    else:
        run_compare(["BTCUSD", "ETHUSD", "SOLUSD", "HYPEUSD", "XRPUSD"], [60, 240])
    return 0


if __name__ == "__main__":
    sys.exit(main())
