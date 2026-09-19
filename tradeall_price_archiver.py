#!/usr/bin/env python3
"""
tradeall_price_archiver.py — captures the LIVE price (from the same public WS
stream as tradeall.py) in a SEPARATE cache24 with LONG retention (12 months by
default, controlled by --months) instead of 24 hours. From now on, it builds a
DENSE-resolution price history (~1 second, like live data) for future backtests
that are much more faithful than cache_price_{symbol}.jsonl (the existing,
significantly sparser history; see the caveat in plan section A5).

It DOES NOT touch tradeall.py and DOES NOT write to
cache_24price_{symbol}.json (the LIVE file used by tradeall for decisions). It
writes separately to cachedb/cache_24price_long_{symbol}.jsonl. This is a
SEPARATE process with its own WS connection (public market stream, no API keys),
so it is safe to run in parallel with tradeall.py.

July 21: uses Cache24LongPriceManager (cacheManager.py), a class DEDICATED to
this script and separate from Cache24PriceManager (used by tradeall.py and left
completely untouched). It persists JSONL incrementally instead of rewriting a
complete JSON document on every save. The archive had reached ~20 MB per symbol
and continues toward several hundred MB at the six-month target; the cost of a
full rewrite grew with the archive, while JSONL writes only new ticks.

Run it continuously, like tradeall.py itself:
    ./tradeall_price_archiver.py --symbols BTCUSDC,TAOUSDC --months 12

After enough dense history has accumulated:
    ./offline/backtests/tradeall.py --symbol BTCUSDC --start <data> --source cache24 \\
        --cache24-file cachedb/cache_24price_long_BTCUSDC.jsonl
"""
import argparse
import math
import os
import re
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import cacheManager as cm
from binance_api import bapi_ws
from botcore import single_instance
from instrument_registry import symbols_for

CACHEDB_DIR = os.path.join(ROOT, "cachedb")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,24}$")


def _positive_finite(value, name, *, minimum=0.001):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _symbols(raw):
    values = []
    for item in str(raw).split(","):
        symbol = item.strip().upper()
        if not symbol:
            continue
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError(f"invalid symbol: {symbol!r}")
        if symbol not in values:
            values.append(symbol)
    if not values:
        raise ValueError("the symbol list cannot be empty")
    return values


def _shutdown(caches, current_price_mgr, ws_module=bapi_ws):
    """Stop producers, flush every pending tick, then close shared resources."""
    for cache in caches:
        current_price_mgr.unsubscribe_price(cache)
    for cache in caches:
        cache.shutdown()
        cache.save_state_to_file()
    current_price_mgr.shutdown()
    ws_module.bapi_ws_manager.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default=None,
                    help="explicit override; otherwise use instruments.conf role.archive")
    p.add_argument("--months", type=float, default=cm.CM_LONG_ARCHIVE_MONTHS,
                    help="retention in months (default from config.env)")
    p.add_argument("--sync-ts", type=float, default=cm.CM_LONG_ARCHIVE_SAMPLE_SEC,
                    help="nominal sampling cadence (default from config.env)")
    p.add_argument("--save-every", type=float, default=cm.CM_LONG_ARCHIVE_FLUSH_SEC,
                    help="disk flush cadence (default from config.env)")
    args = p.parse_args()
    try:
        symbols = _symbols(args.symbols if args.symbols is not None
                           else ",".join(symbols_for("binance", "archive")))
        months = _positive_finite(args.months, "--months")
        sync_ts = _positive_finite(args.sync_ts, "--sync-ts", minimum=0.1)
        save_every = _positive_finite(args.save_every, "--save-every", minimum=1.0)
    except (TypeError, ValueError) as exc:
        p.error(str(exc))
    keep_hours = months * 30 * 24

    single_instance("tradeall_price_archiver")
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        print(f"[tradeall_price_archiver] signal {signum}; flushing and stopping...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    os.makedirs(CACHEDB_DIR, exist_ok=True)
    # This process consumes only the public market stream. The private account stream
    # belongs to the dedicated cache process and would start redundant order/trade caches.
    current_price_mgr = cm.get_current_price_manager(
        ws_manager=bapi_ws.get_ws_manager(),
        sync_ts=sync_ts,
    )

    caches = []
    for symbol in symbols:
        filename = os.path.join(CACHEDB_DIR, f"cache_24price_long_{symbol}.jsonl")
        cache = cm.Cache24LongPriceManager(sync_ts=save_every, symbols=[symbol], filename=filename)
        cache.KEEP_HOURS = keep_hours   # per-instance override explicitly supported by the class
        cache.RETENTION_DAYS = keep_hours / 24.0
        cache.enable_save_state_to_file()   # defaults to False; required for disk writes
        cache.periodic_sync(save_every, save_state=True)
        current_price_mgr.subscribe_price(cache)
        caches.append(cache)

    print(f"[tradeall_price_archiver] started: {symbols} | retention {months} months "
          f"({keep_hours:.0f}h) | disk flush every {save_every:.0f}s", flush=True)
    print("[tradeall_price_archiver] writing cachedb/cache_24price_long_<symbol>.jsonl "
          "separately from tradeall.py's live 24-hour cache")
    print("[tradeall_price_archiver] Ctrl+C stops the process.")

    try:
        while not stop_event.wait(60):
            print("[tradeall_price_archiver] heartbeat", flush=True)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        _shutdown(caches, current_price_mgr)
        print("\n[tradeall_price_archiver] stopped after the flush.", flush=True)


if __name__ == "__main__":
    main()
