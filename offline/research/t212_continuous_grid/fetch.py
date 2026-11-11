#!/usr/bin/env python3
"""Fetch Yahoo 1h (last ~730 days) and 1d (10 years) regular-session bars for the grid."""
import argparse, os, sys, time
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from botcore import optional_env
from offline.backtests.datasets import load_ohlc_tuples_csv, save_dataset, validate_dataset
from providers.market_api import api as mkt


def fetch_yahoo_candles(sym: str, *, range_: str = "730d", interval: str = "1h", drop_incomplete: bool = True) -> list[dict]:
    return mkt.fetch_ohlc_candles(
        sym,
        interval=interval,
        range_=range_,
        drop_incomplete=drop_incomplete,
        provider_name="t212",
    )

DATA = os.path.join(optional_env("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid"), "data")
ASSETS = ["NVDA", "SPCX", "RGNT", "AAPL", "MSFT", "AMD", "TSLA", "AVGO", "META", "AMZN",
          "GOOGL", "NFLX", "PLTR", "COIN", "QQQ", "SPY"]


def fetch(sym, rng, itv):
    candles = fetch_yahoo_candles(sym, range_=rng, interval=itv, drop_incomplete=True)
    if not candles:
        return []
    validated = validate_dataset(candles)
    return [(c["timestamp"], c["open"], c["high"], c["low"], c["close"]) for c in validated]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--missing-only", action="store_true", help="skip symbols with an existing >32-bar CSV")
    ap.add_argument("--force", action="store_true", help="re-fetch all symbols regardless of existing CSVs")
    cli_args = ap.parse_args()
    missing_only = bool(cli_args.missing_only and not cli_args.force)
    os.makedirs(DATA, exist_ok=True)
    usable_total = 0
    for sym in ASSETS:
        for rng, itv in (("730d", "1h"), ("10y", "1d")):
            csv_path = os.path.join(DATA, f"{sym}_{itv}.csv")
            has_cached_csv = os.path.exists(csv_path) and len(load_ohlc_tuples_csv(csv_path, strict=False)) > 32
            if missing_only and has_cached_csv:
                usable_total += 1
                continue
            try:
                candles = fetch_yahoo_candles(sym, range_=rng, interval=itv, drop_incomplete=True)
                if not candles:
                    print(f"[fetch] {sym} {itv}: no bars returned", flush=True)
                    if has_cached_csv:
                        usable_total += 1
                    continue
                validated = validate_dataset(candles)
                if len(validated) <= 32:
                    print(
                        f"[fetch] {sym} {itv}: insufficient bars ({len(validated)} <= 32) -- refusing overwrite",
                        flush=True,
                    )
                    if has_cached_csv:
                        usable_total += 1
                    continue
                save_dataset(validated, Path(csv_path))
                usable_total += 1
                print(f"[fetch] {sym} {itv}: {len(validated)} bars "
                      f"{time.strftime('%Y-%m-%d', time.gmtime(validated[0]['timestamp']))} -> "
                      f"{time.strftime('%Y-%m-%d', time.gmtime(validated[-1]['timestamp']))}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] {sym} {itv}: {exc}", flush=True)
                if has_cached_csv:
                    usable_total += 1
            finally:
                time.sleep(0.5)
    if usable_total == 0:
        print("[fetch] ERROR: zero usable Yahoo candle CSVs available", file=sys.stderr, flush=True)
        sys.exit(1)
