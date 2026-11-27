#!/usr/bin/env python3
"""Fetch Yahoo 1h (last ~730 days) and 1d (10 years) regular-session bars for the grid."""
import os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from botcore import optional_env
from offline.backtests.datasets import (
    ohlc_range_window_seconds,
    parse_ohlc_refresh_cli_args,
    refresh_ohlc_csv_dataset,
    validate_dataset,
)
from providers.market_api import api as mkt
from strategies.replay_metrics import ohlc_records_to_timed_tuples


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
    return ohlc_records_to_timed_tuples(validate_dataset(candles)) if candles else []


MIN_GRID_BARS = 33  # len(rows) > 32 (WARM + 12)


def _fetch_paced(sym: str, rng: str, itv: str) -> list[dict]:
    try:
        return fetch_yahoo_candles(sym, range_=rng, interval=itv, drop_incomplete=True)
    finally:
        time.sleep(0.5)


if __name__ == "__main__":
    force_refresh, missing_only, max_age_sec = parse_ohlc_refresh_cli_args(__doc__)
    os.makedirs(DATA, exist_ok=True)
    usable_total = sum(
        1
        for sym in ASSETS
        for rng, itv in (("730d", "1h"), ("10y", "1d"))
        if refresh_ohlc_csv_dataset(
            os.path.join(DATA, f"{sym}_{itv}.csv"),
            lambda s=sym, r=rng, i=itv: _fetch_paced(s, r, i),
            label=f"{sym} {itv}",
            min_bars=MIN_GRID_BARS,
            strict=False,
            force=force_refresh,
            missing_only=missing_only,
            max_age_sec=max_age_sec,
            window_seconds=ohlc_range_window_seconds(rng),
            log_skip=False,
        )
    )
    if usable_total == 0:
        print("[fetch] ERROR: zero usable Yahoo candle CSVs available", file=sys.stderr, flush=True)
        sys.exit(1)
