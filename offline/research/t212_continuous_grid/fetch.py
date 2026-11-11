#!/usr/bin/env python3
"""Fetch Yahoo 1h (last ~730 days) and 1d (10 years) regular-session bars for the grid."""
import csv, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from providers.market_api import api as mkt
from state_io import atomic_text_writer


def fetch_yahoo_candles(sym: str, *, range_: str = "730d", interval: str = "1h", drop_incomplete: bool = True) -> list[dict]:
    return mkt.fetch_ohlc_candles(
        sym,
        interval=interval,
        range_=range_,
        drop_incomplete=drop_incomplete,
        provider_name="t212",
    )

DATA = os.path.join(os.environ.get("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid"), "data")
ASSETS = ["NVDA", "SPCX", "RGNT", "AAPL", "MSFT", "AMD", "TSLA", "AVGO", "META", "AMZN",
          "GOOGL", "NFLX", "PLTR", "COIN", "QQQ", "SPY"]


def fetch(sym, rng, itv):
    candles = fetch_yahoo_candles(sym, range_=rng, interval=itv, drop_incomplete=True)
    return [(c["timestamp"], c["open"], c["high"], c["low"], c["close"]) for c in candles]


if __name__ == "__main__":
    missing_only = "--missing-only" in sys.argv[1:]
    os.makedirs(DATA, exist_ok=True)
    attempted = 0
    succeeded = 0
    for sym in ASSETS:
        for rng, itv in (("730d", "1h"), ("10y", "1d")):
            csv_path = os.path.join(DATA, f"{sym}_{itv}.csv")
            if missing_only and os.path.exists(csv_path) and os.path.getsize(csv_path) > 64:
                continue
            attempted += 1
            try:
                rows = fetch(sym, rng, itv)
                if not rows:
                    print(f"[fetch] {sym} {itv}: no bars returned", flush=True)
                    continue
                with atomic_text_writer(csv_path, newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["timestamp", "open", "high", "low", "close"])
                    w.writerows(rows)
                succeeded += 1
                print(f"[fetch] {sym} {itv}: {len(rows)} bars "
                      f"{time.strftime('%Y-%m-%d', time.gmtime(rows[0][0]))} -> "
                      f"{time.strftime('%Y-%m-%d', time.gmtime(rows[-1][0]))}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] {sym} {itv}: {exc}", flush=True)
            finally:
                time.sleep(0.5)
    if attempted > 0 and succeeded == 0:
        print("[fetch] ERROR: all requested Yahoo candle fetches failed", file=sys.stderr, flush=True)
        sys.exit(1)
