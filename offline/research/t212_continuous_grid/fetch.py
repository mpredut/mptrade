#!/usr/bin/env python3
"""Fetch Yahoo 1h (last ~730 days) and 1d (10 years) regular-session bars for the grid."""
import os, sys, time
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from botcore import optional_env
from offline.backtests.datasets import (
    format_ohlc_bars_log_span,
    merge_with_existing_ohlc_csv,
    ohlc_range_window_seconds,
    parse_ohlc_refresh_cli_args,
    save_dataset,
    should_skip_ohlc_csv_refresh,
    validate_dataset,
)
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


MIN_GRID_BARS = 33  # len(rows) > 32 (WARM + 12)


if __name__ == "__main__":
    force_refresh, missing_only, max_age_sec = parse_ohlc_refresh_cli_args(__doc__)
    os.makedirs(DATA, exist_ok=True)
    usable_total = 0
    for sym in ASSETS:
        for rng, itv in (("730d", "1h"), ("10y", "1d")):
            csv_path = os.path.join(DATA, f"{sym}_{itv}.csv")
            skip, has_cached_csv = should_skip_ohlc_csv_refresh(
                csv_path,
                min_bars=MIN_GRID_BARS,
                strict=False,
                force=force_refresh,
                missing_only=missing_only,
                max_age_sec=max_age_sec,
            )
            if skip:
                usable_total += 1
                continue
            try:
                candles = fetch_yahoo_candles(sym, range_=rng, interval=itv, drop_incomplete=True)
                if not candles:
                    print(f"[fetch] {sym} {itv}: no bars returned", flush=True)
                    if has_cached_csv:
                        usable_total += 1
                    continue
                merged = merge_with_existing_ohlc_csv(
                    candles,
                    csv_path,
                    window_seconds=ohlc_range_window_seconds(rng),
                )
                if not (len(merged) > 32):
                    print(
                        f"[fetch] {sym} {itv}: insufficient bars ({len(merged)} <= 32) -- refusing overwrite",
                        flush=True,
                    )
                    if has_cached_csv:
                        usable_total += 1
                    continue
                save_dataset(merged, Path(csv_path))
                usable_total += 1
                print(f"[fetch] {sym} {itv}: {format_ohlc_bars_log_span(merged)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] {sym} {itv}: {exc}", flush=True)
                if has_cached_csv:
                    usable_total += 1
            finally:
                time.sleep(0.5)
    if usable_total == 0:
        print("[fetch] ERROR: zero usable Yahoo candle CSVs available", file=sys.stderr, flush=True)
        sys.exit(1)
