#!/usr/bin/env python3
"""
analyze_volatility.py — analyze an asset's Yahoo OHLC volatility to CALIBRATE
t212_bot DCA spacing, take profit, and stop loss from data rather than guesses.
Reusable for any new asset.

  python3 analyze_volatility.py RGNT
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import urllib.parse
import urllib.request


def fetch_ohlc(sym: str, rng: str, interval: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(str(sym or ''))}?range={rng}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.loads(r.read())
    chart = data.get("chart") if isinstance(data, dict) else None
    chart = chart if isinstance(chart, dict) else {}
    res = (chart.get("result") or [None])[0]
    if not isinstance(res, dict) or not res:
        return [], {}
    ts = res.get("timestamp") or []
    indicators = res.get("indicators") if isinstance(res.get("indicators"), dict) else {}
    quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
    q = (quote_list[0] if quote_list and isinstance(quote_list[0], dict) else {}) or {}
    o, h, l, c, v = (q.get(k) or [] for k in ("open", "high", "low", "close", "volume"))
    n_ohlc = min(len(o), len(h), len(l), len(c))
    out = []
    for i, t in enumerate(ts):
        if i < n_ohlc and None not in (o[i], h[i], l[i], c[i]):
            if any(isinstance(val, bool) for val in (o[i], h[i], l[i], c[i])):
                continue
            try:
                of, hf, lf, cf = float(o[i]), float(h[i]), float(l[i]), float(c[i])
                vf = float(v[i]) if i < len(v) and v[i] is not None and not isinstance(v[i], bool) else 0.0
            except (TypeError, ValueError, OverflowError):
                continue
            if all(math.isfinite(val) and val > 0 for val in (of, hf, lf, cf)):
                out.append((t, of, hf, lf, cf, vf if math.isfinite(vf) and vf > 0 else 0))
    meta = res.get("meta")
    return out, meta if isinstance(meta, dict) else {}


def pct(a, b):
    return (a - b) / b * 100 if b else 0.0


def pullbacks(closes):
    """Return pullback depths below the running peak for DCA spacing."""
    if not closes:
        return []
    peak = closes[0]; dips = []; cur = 0.0
    for c in closes:
        if c >= peak:
            if cur > 0:
                dips.append(cur)
            peak = c; cur = 0.0
        else:
            cur = max(cur, pct(peak, c))
    if cur > 0:
        dips.append(cur)
    return dips


def main() -> int:
    sym = sys.argv[1] if len(sys.argv) > 1 else "RGNT"
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

    print(f"========== VOLATILITATE {sym} ==========")
    print(f"  current price {last:.2f} {meta.get('currency','')}  | exchange {meta.get('exchangeName','?')}")
    print(f"  AZI: open {o0:.2f} -> last {last:.2f}  ({pct(last,o0):+.1f}%)  | range zi {pct(hi,lo):.1f}% (lo {lo:.2f} / hi {hi:.2f})")
    print(f"  spike max (lo->hi azi): {pct(hi,lo):+.1f}%")
    print(f"\n  --- bare 5min ({len(intra)} bare) ---")
    print(f"  miscare medie/bar : {statistics.mean(abs_rets):.2f}%   (mediana {statistics.median(abs_rets):.2f}%)")
    print(f"  bar 5m p90        : {sorted(abs_rets)[int(len(abs_rets)*0.9)]:.2f}%   max {max(abs_rets):.2f}%")
    print(f"  range mediu/bar   : {statistics.mean(bar_rng):.2f}%  (ATR-5m aprox)")
    if dips:
        ds = sorted(dips)
        print(f"\n  --- retrageri (dip-uri) azi: {len(dips)} ---")
        print(f"  dip median {statistics.median(dips):.2f}%  | p75 {ds[int(len(ds)*0.75)]:.2f}%  | max {max(dips):.2f}%")
    if daily:
        dr = [pct(b[2], b[3]) for b in daily[-10:]]
        dchg = [abs(pct(b[4], b[1])) for b in daily[-10:]]
        print(f"\n  --- zilnic (ultimele {len(dr)} zile) ---")
        print(f"  range mediu/zi {statistics.mean(dr):.1f}%  | miscare medie close-open {statistics.mean(dchg):.1f}%  | max zi {max(dr):.1f}%")

    # --- PARAMETER SUGGESTION using transparent heuristics ---
    fee = 0.30  # FX 0.15%x2
    med_dip = statistics.median(dips) if dips else statistics.mean(abs_rets) * 2
    atr5 = statistics.mean(bar_rng)
    tp = max(round(med_dip * 1.3, 1), round(fee + 2 * atr5, 1))   # clear fees/noise and capture a real swing
    drop = round(max(med_dip, atr5 * 1.5), 1)                     # DCA at a typical dip
    daily_rng = statistics.mean([pct(b[2], b[3]) for b in daily[-10:]]) if daily else 10
    stop = round(min(max(daily_rng * 1.3, tp * 2.5), 25), 0)      # above a typical daily swing, capped at 25%
    print(f"\n  ===> SUGESTIE (euristic, pt budget $3000):")
    print(f"       TP            = {tp}%   (a fee of {fee}% plus 2xATR5 {2*atr5:.1f}%, or 1.3x the median dip)")
    print(f"       DCA_DROP_PCT  = {drop}%")
    print(f"       STOP_LOSS_PCT = {stop:.0f}%")
    print("=========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
