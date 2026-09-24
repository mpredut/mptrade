#!/usr/bin/env python3
"""Fetch Yahoo 1h (last ~730 days) and 1d (10 years) regular-session bars for the grid."""
import csv, json, os, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(os.environ.get("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid"), "data")
ASSETS = ["NVDA", "SPCX", "RGNT", "AAPL", "MSFT", "AMD", "TSLA", "AVGO", "META", "AMZN",
          "GOOGL", "NFLX", "PLTR", "COIN", "QQQ", "SPY"]


def fetch(sym, rng, itv):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval={itv}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    res = json.loads(urllib.request.urlopen(req, timeout=30).read())["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = [(t, o, h, l, c) for t, o, h, l, c in
            zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"])
            if None not in (o, h, l, c)]
    return rows[:-1]   # drop the forming bar


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for sym in ASSETS:
        for rng, itv in (("730d", "1h"), ("10y", "1d")):
            try:
                rows = fetch(sym, rng, itv)
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] {sym} {itv}: {exc}")
                continue
            with open(os.path.join(DATA, f"{sym}_{itv}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["timestamp", "open", "high", "low", "close"])
                w.writerows(rows)
            print(f"[fetch] {sym} {itv}: {len(rows)} bars "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(rows[0][0]))} -> "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(rows[-1][0]))}", flush=True)
            time.sleep(0.5)
