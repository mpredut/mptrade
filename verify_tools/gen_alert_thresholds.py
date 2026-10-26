#!/usr/bin/env python3
"""
    gen_alert_thresholds.py — generates alert thresholds (UP/DOWN) for coins from
    their actual VOLATILITY (CoinGecko), not from a blind default. (Formerly suggest_thresholds.py.)

    Logic: at every point, calculate the same measures as PriceChecker: "rise from the
    24-hour low" and "drop from the 24-hour high". Use the p85 percentile of these
    movements, so the threshold captures high-movement parts of days rather than normal daily noise.

    Output consists of lines ready for market_alerts.conf. Review them manually; they
    are provisional because a new coin has sparse history and improves as data accumulates.

  python3 verify_tools/gen_alert_thresholds.py SPCXX TAO BTC
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import urllib.request

WINDOW = 24       # Approximate hourly observations in a 24-hour window (CoinGecko hourly granularity).
PERCENTILE = 85   # Movement at or above this percentile is considered notable.
DAYS = 14
FLOOR = 3.0       # Do not lower the threshold below this value, or noise will trigger alerts.


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def resolve_id(symbol):
    """Map a symbol to (coingecko_id, symbol, name), preferring an exact symbol match."""
    coins = _get(f"https://api.coingecko.com/api/v3/search?query={symbol}").get("coins", [])
    for c in coins:
        if c["symbol"].upper() == symbol.upper():
            return c["id"], c["symbol"].upper(), c.get("name", "")
    if coins:
        c = coins[0]
        return c["id"], c["symbol"].upper(), c.get("name", "")
    return None, None, None


def _perc(xs, pct):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * pct / 100))]


def thresholds_from_prices(prices, window=WINDOW, pct=PERCENTILE):
    """PURE (testable): from the price series -> (up, down, up_median, down_median)."""
    if (
        not prices
        or isinstance(window, bool)
        or isinstance(pct, bool)
    ):
        return None
    try:
        win_i = int(window)
        pct_f = float(pct)
    except (TypeError, ValueError):
        return None
    if win_i <= 0 or not math.isfinite(pct_f) or not (0.0 < pct_f <= 100.0):
        return None
    clean_prices = []
    for p in prices:
        if p is None or isinstance(p, bool):
            continue
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if math.isfinite(pf) and pf > 0:
            clean_prices.append(pf)
    ups, downs = [], []
    for i in range(win_i, len(clean_prices)):
        win = clean_prices[i - win_i:i + 1]
        mn, mx, cur = min(win), max(win), clean_prices[i]
        if mn > 0:
            ups.append((cur - mn) / mn * 100)
        if mx > 0:
            downs.append((mx - cur) / mx * 100)
    if not ups or not downs:
        return None
    return (
        max(round(_perc(ups, pct_f), 1), FLOOR),
        max(round(_perc(downs, pct_f), 1), FLOOR),
        round(statistics.median(ups), 1),
        round(statistics.median(downs), 1),
    )


def suggest(symbol, days=DAYS):
    cid, sym, name = resolve_id(symbol)
    if not cid:
        return None
    data = _get(f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart?vs_currency=usd&days={days}")
    prices = [p[1] for p in data.get("prices", []) if p and p[1]]
    if len(prices) < WINDOW + 5:
        return None
    res = thresholds_from_prices(prices)
    if not res:
        return None
    up, down, up_med, down_med = res
    return {"symbol": sym, "name": name, "up": up, "down": down,
            "up_median": up_med, "down_median": down_med, "points": len(prices)}


def main() -> int:
    syms = sys.argv[1:] or ["BTC", "TAO"]
    print(f"# thresholds suggested from volatility (p{PERCENTILE} over {DAYS} days) — REVIEW, then put them in market_alerts.conf")
    for s in syms:
        try:
            r = suggest(s)
        except Exception as e:  # noqa: BLE001
            print(f"# {s}: eroare ({e})"); continue
        if not r:
            print(f"# {s}: date insuficiente / negasit pe CoinGecko"); continue
        print(f"{r['symbol']:<8} = {r['up']} / {r['down']}    "
              f"# {r['name']} | tipic zilnic +{r['up_median']}/-{r['down_median']}% ({r['points']} puncte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
