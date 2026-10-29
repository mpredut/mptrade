#!/usr/bin/env python3
"""
analyze_volatility.py — analyze an asset's Yahoo OHLC volatility to CALIBRATE
t212_bot DCA spacing, take profit, and stop loss from data rather than guesses.
Reusable for any new asset.

  python3 analyze_volatility.py RGNT
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import safe_float, safe_pos_float


def fetch_ohlc(sym: str, rng: str, interval: str):
    q_sym = urllib.parse.quote(str(sym or "").strip(), safe="")
    q_rng = urllib.parse.quote(str(rng or "").strip(), safe="")
    q_int = urllib.parse.quote(str(interval or "").strip(), safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{q_sym}?range={q_rng}&interval={q_int}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return [], {}
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
        if i < n_ohlc and None not in (t, o[i], h[i], l[i], c[i]):
            tf = safe_pos_float(t, default=None)
            of = safe_pos_float(o[i], default=None)
            hf = safe_pos_float(h[i], default=None)
            lf = safe_pos_float(l[i], default=None)
            cf = safe_pos_float(c[i], default=None)
            if tf is None or of is None or hf is None or lf is None or cf is None:
                continue
            if not (hf >= max(of, cf, lf) and lf <= min(of, cf, hf)):
                continue
            raw_v = v[i] if i < len(v) else 0.0
            vf = safe_pos_float(raw_v, default=0.0)
            out.append((int(tf) if isinstance(t, int) else tf, of, hf, lf, cf, vf))
    meta = res.get("meta")
    return out, meta if isinstance(meta, dict) else {}


def pct(a, b):
    af = safe_float(a, default=None)
    bf = safe_pos_float(b, default=None)
    if af is None or bf is None:
        return 0.0
    return (af - bf) / bf * 100.0


def pullbacks(closes):
    """Return pullback depths below the running peak for DCA spacing."""
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
    print(f"  miscare medie/bar : {statistics.fmean(abs_rets):.2f}%   (mediana {statistics.median(abs_rets):.2f}%)")
    print(f"  bar 5m p90        : {sorted(abs_rets)[min(len(abs_rets) - 1, int(len(abs_rets)*0.9))]:.2f}%   max {max(abs_rets):.2f}%")
    print(f"  range mediu/bar   : {statistics.fmean(bar_rng):.2f}%  (ATR-5m aprox)")
    if dips:
        ds = sorted(dips)
        print(f"\n  --- retrageri (dip-uri) azi: {len(dips)} ---")
        print(f"  dip median {statistics.median(dips):.2f}%  | p75 {ds[min(len(ds) - 1, int(len(ds)*0.75))]:.2f}%  | max {max(dips):.2f}%")
    if daily:
        dr = [pct(b[2], b[3]) for b in daily[-10:]]
        dchg = [abs(pct(b[4], b[1])) for b in daily[-10:]]
        print(f"\n  --- zilnic (ultimele {len(dr)} zile) ---")
        print(f"  range mediu/zi {statistics.fmean(dr):.1f}%  | miscare medie close-open {statistics.fmean(dchg):.1f}%  | max zi {max(dr):.1f}%")

    # --- PARAMETER SUGGESTION using transparent heuristics ---
    fee = 0.30  # FX 0.15%x2
    med_dip = statistics.median(dips) if dips else statistics.fmean(abs_rets) * 2
    atr5 = statistics.fmean(bar_rng)
    tp = max(round(med_dip * 1.3, 1), round(fee + 2 * atr5, 1))   # clear fees/noise and capture a real swing
    drop = round(max(med_dip, atr5 * 1.5), 1)                     # DCA at a typical dip
    daily_rng = statistics.fmean([pct(b[2], b[3]) for b in daily[-10:]]) if daily else 10
    stop = round(min(max(daily_rng * 1.3, tp * 2.5), 25), 0)      # above a typical daily swing, capped at 25%
    print(f"\n  ===> SUGESTIE (euristic, pt budget $3000):")
    print(f"       TP            = {tp}%   (a fee of {fee}% plus 2xATR5 {2*atr5:.1f}%, or 1.3x the median dip)")
    print(f"       DCA_DROP_PCT  = {drop}%")
    print(f"       STOP_LOSS_PCT = {stop:.0f}%")
    print("=========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
