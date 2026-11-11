#!/usr/bin/env python3
"""
analyze_volatility.py -- analyze an asset's Yahoo OHLC volatility to CALIBRATE
t212_bot DCA spacing, take profit, and stop loss from data rather than guesses.
Reusable for any new asset.

  python3 analyze_volatility.py RGNT
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    ensure_local_module_priority,
    exact_pos_int,
    fetch_public_json,
    float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
ensure_local_module_priority(_HERE, "ipo_common", "market_data")
from cachepaths import has_equity_suffix, strip_equity_suffix
from ipo_common import load_isolated_t212_cfg as _load_isolated_t212_cfg  # noqa: E402
from market_data import parse_interval_minutes  # noqa: E402
from providers.order_ohlc_utils import (
    drop_incomplete_last_bar,
    extract_yahoo_chart_result_and_quote as _extract_yahoo_result_and_quote,
)


def _quantile_sorted(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def fetch_ohlc(
    sym: str,
    rng: str,
    interval: str,
    *,
    drop_incomplete: bool = True,
    now_timestamp: int | None = None,
):
    if isinstance(sym, bool) or not isinstance(sym, str) or not sym.strip():
        return [], {}
    raw_sym = sym.strip()
    norm_sym = strip_equity_suffix(raw_sym) if has_equity_suffix(raw_sym) else raw_sym
    q_sym = urllib.parse.quote(norm_sym, safe="")
    q_rng = urllib.parse.quote(str(rng or "").strip(), safe="")
    q_int = urllib.parse.quote(str(interval or "").strip(), safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{q_sym}?range={q_rng}&interval={q_int}"
    try:
        data = fetch_public_json(url, timeout=25.0, user_agent="Mozilla/5.0")
    except (urllib.error.URLError, OSError, ValueError):
        return [], {}
    res, _, ts, q = _extract_yahoo_result_and_quote(data)
    if res is None:
        return [], {}
    o, h, l, c, v = (q.get(k) or [] for k in ("open", "high", "low", "close", "volume"))
    n_ohlc = min(len(o), len(h), len(l), len(c))
    by_ts: dict[int, dict] = {}
    for i, t in enumerate(ts):
        if i < n_ohlc and None not in (t, o[i], h[i], l[i], c[i]):
            ts_i = exact_pos_int(t, default=None)
            of = safe_pos_float(o[i], default=None)
            hf = safe_pos_float(h[i], default=None)
            lf = safe_pos_float(l[i], default=None)
            cf = safe_pos_float(c[i], default=None)
            if ts_i is None or of is None or hf is None or lf is None or cf is None:
                continue
            if not (hf >= max(of, cf, lf) and lf <= min(of, cf, hf)):
                continue
            raw_v = v[i] if i < len(v) else 0.0
            vf = safe_nonneg_float(raw_v, default=0.0)
            by_ts[ts_i] = {"timestamp": ts_i, "open": of, "high": hf, "low": lf, "close": cf, "volume": vf}
    records = [by_ts[k] for k in sorted(by_ts)]
    vol_by_ts = {r["timestamp"]: r["volume"] for r in records}
    meta = res.get("meta")
    meta_d = meta if isinstance(meta, dict) else {}
    if drop_incomplete and records:
        try:
            iv_min = parse_interval_minutes(str(interval or "").strip())
            regular = ((meta_d.get("currentTradingPeriod") or {}).get("regular")) or {}
            eff_now = exact_pos_int(now_timestamp, default=None)
            if eff_now is None:
                eff_now = exact_pos_int(math.floor(time.time()), default=1) or 1
            records = drop_incomplete_last_bar(
                records,
                interval_minutes=iv_min,
                now_timestamp=eff_now,
                regular_session_start=exact_pos_int(regular.get("start"), default=None),
                regular_session_end=exact_pos_int(regular.get("end"), default=None),
            )
        except ValueError:
            pass
    out = [
        (r["timestamp"], r["open"], r["high"], r["low"], r["close"], vol_by_ts.get(r["timestamp"], 0.0))
        for r in records
    ]
    return out, meta_d


def pct(a, b):
    af = safe_float(a, default=None)
    bf = safe_pos_float(b, default=None)
    if af is None or bf is None:
        return 0.0
    return (af - bf) / bf * 100.0


def pullbacks(closes, *, include_ongoing: bool = False):
    """Return completed pullback depths below the running peak for DCA spacing."""
    if closes is None or isinstance(closes, bool):
        return []
    try:
        clean = [cf for c in closes if (cf := safe_pos_float(c, default=None)) is not None]
    except TypeError:
        return []
    if not clean:
        return []
    peak = clean[0]; dips = []; cur = 0.0
    for c in clean:
        if c >= peak:
            if cur > 0:
                dips.append(cur)
            peak = c; cur = 0.0
        else:
            cur = max(cur, (peak - c) / peak * 100.0)
    if include_ongoing and cur > 0:
        dips.append(cur)
    return dips


def _load_t212_cfg(profile: str | None = None) -> dict[str, str]:
    prof_name = profile if profile else "nvda"
    prof_path = os.path.join(_HERE, f"config.{prof_name}.env")
    if not os.path.exists(prof_path):
        prof_path = os.path.join(_HERE, "config.env")
    return _load_isolated_t212_cfg(
        os.path.join(_HERE, ".env"),
        profile_file=prof_path if os.path.exists(prof_path) else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze Yahoo OHLC volatility to calibrate t212_bot parameters.")
    ap.add_argument("symbol", nargs="?", default=None, help="Yahoo Finance ticker symbol (e.g. NVDA, RGNT)")
    ap.add_argument("--profile", default=None, help="Optional T212 config.<profile>.env name")
    args = ap.parse_args(argv)
    if args.profile and (not args.profile.replace("-", "").replace("_", "").isalnum()):
        ap.error(f"invalid profile name: {args.profile!r}")

    cfg = _load_t212_cfg(args.profile)
    sym = (
        args.symbol
        or cfg.get("YAHOO_SYMBOL")
        or cfg.get("STRAT_YAHOO_SYM")
        or cfg.get("T212_TICKER")
        or "NVDA"
    ).strip()
    intra, meta = fetch_ohlc(sym, "1d", "5m")
    daily, _ = fetch_ohlc(sym, "1mo", "1d")
    if not intra:
        print(f"! no intraday data for {sym} (a wrong Yahoo symbol?)"); return 1

    o0 = intra[0][1]; last = intra[-1][4]
    hi = max(b[2] for b in intra); lo = min(b[3] for b in intra)
    closes = [b[4] for b in intra]
    rets = [pct(closes[i], closes[i - 1]) for i in range(1, len(closes))]
    abs_rets = [abs(r) for r in rets] or [0.0]
    bar_rng = [pct(b[2], b[3]) for b in intra]           # (high-low)/low per five-minute bar
    dips = pullbacks(closes)

    print(f"========== VOLATILITY {sym} ==========")
    print(f"  current price {last:.2f} {meta.get('currency','')}  | exchange {meta.get('exchangeName','?')}")
    print(f"  TODAY: open {o0:.2f} -> last {last:.2f}  ({pct(last,o0):+.1f}%)  | day range {pct(hi,lo):.1f}% (lo {lo:.2f} / hi {hi:.2f})")
    print(f"  max spike (lo->hi today): {pct(hi,lo):+.1f}%")
    print(f"\n  --- 5min bars ({len(intra)} bars) ---")
    print(f"  mean move/bar     : {statistics.fmean(abs_rets):.2f}%   (median {statistics.median(abs_rets):.2f}%)")
    print(f"  bar 5m p90        : {_quantile_sorted(sorted(abs_rets), 0.9):.2f}%   max {max(abs_rets):.2f}%")
    print(f"  mean range/bar    : {statistics.fmean(bar_rng):.2f}%  (approx ATR-5m)")
    if dips:
        ds = sorted(dips)
        print(f"\n  --- pullbacks (dips) today: {len(dips)} ---")
        print(f"  median dip {statistics.median(dips):.2f}%  | p75 {_quantile_sorted(ds, 0.75):.2f}%  | max {max(dips):.2f}%")
    if daily:
        dr = [pct(b[2], b[3]) for b in daily[-10:]]
        dchg = [abs(pct(b[4], b[1])) for b in daily[-10:]]
        print(f"\n  --- daily (last {len(dr)} days) ---")
        print(f"  mean range/day {statistics.fmean(dr):.1f}%  | mean close-open move {statistics.fmean(dchg):.1f}%  | max day {max(dr):.1f}%")

    # --- PARAMETER SUGGESTION using transparent heuristics ---
    fx_fee = float_env("STRAT_FX_FEE_PCT", cfg)
    if fx_fee is None:
        fx_fee = float_env("T212_FX_FEE_PCT", cfg)
    fee = round(2.0 * (fx_fee if fx_fee is not None else 0.15), 4)
    budget_ref = float_env("STRAT_MAX_BUDGET", cfg) or float_env("STRAT_TOTAL_BUDGET", cfg) or 3000.0
    med_dip = statistics.median(dips) if dips else statistics.fmean(abs_rets) * 2
    atr5 = statistics.fmean(bar_rng)
    tp = max(round(med_dip * 1.3, 1), round(fee + 2 * atr5, 1))   # clear fees/noise and capture a real swing
    drop = round(max(med_dip, atr5 * 1.5), 1)                     # DCA at a typical dip
    daily_rng = statistics.fmean([pct(b[2], b[3]) for b in daily[-10:]]) if daily else 10
    stop = round(min(max(daily_rng * 1.3, tp * 2.5), 25), 0)      # above a typical daily swing, capped at 25%
    print(f"\n  ===> SUGGESTION (heuristic, for {budget_ref:g} budget):")
    print(f"       TP            = {tp}%   (a fee of {fee}% plus 2xATR5 {2*atr5:.1f}%, or 1.3x the median dip)")
    print(f"       DCA_DROP_PCT  = {drop}%")
    print(f"       STOP_LOSS_PCT = {stop:.0f}%")
    print("=========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
