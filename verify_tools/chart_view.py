#!/usr/bin/env python3
"""Read-only on-demand multi-timeframe chart zoom utility for AI executors and operators.

Reads persistent multi-timeframe OHLCV bars from ``cachedb/prices/cache_ohlc_bars_<TF>_<SYM>.json``
(with metadata from ``cachedb/macro/long_horizon_collect.json``, zero network I/O) and renders
an exact ASCII price chart with OHLCV anchors and ZigZag swings for any chosen period
(``--months N`` or ``--from YYYY-MM --to YYYY-MM``) at ``1w``, ``1d``, ``4h``, or ``1h`` resolution.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from botcore import exact_pos_int
from cachepaths import CACHE_LONG_HORIZON_FILENAME, resolve_cachedb_filepath
from intelligence.sentiment.evaluators.long_horizon_evaluator import (
    evaluate_symbol_long_horizon,
    render_ascii_chart_view,
)
from providers.ohlc_bar_archive import aggregate_ohlc_bars, load_ohlc_archive_bars
from state_io import read_json_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect multi-timeframe OHLCV chart slice from persistent cache_ohlc_bars_<TF>_<SYM>.json & long_horizon_collect.json"
    )
    parser.add_argument("symbol", nargs="?", help="Symbol to inspect (e.g. BTCUSDC, HYPEUSDC, NVDA_US_EQ); omit to list available symbols")
    parser.add_argument("--months", default=None, help="Trailing number of months to zoom into (e.g. 1, 6, 12, 18, 24, 36)")
    parser.add_argument("--from", dest="start_ym", default=None, help="Start date YYYY-MM or YYYY-MM-DD (UTC)")
    parser.add_argument("--to", dest="end_ym", default=None, help="End date YYYY-MM or YYYY-MM-DD (UTC)")
    parser.add_argument(
        "--tf",
        choices=("1w", "1d", "4h", "1h"),
        default="1w",
        help="Timeframe: 1w (weekly), 1d (full multi-year daily), 4h (intraday 4h archive), or 1h (intraday 1h archive)",
    )
    parser.add_argument("--width", default="60", help="ASCII chart width in columns")
    parser.add_argument("--height", default="12", help="ASCII chart height in rows")
    parser.add_argument("--cache-dir", default="cachedb", help="Path to cachedb directory")
    args = parser.parse_args(argv)

    months_v = None
    if args.months is not None:
        months_v = exact_pos_int(args.months, default=None)
        if months_v is None:
            print(f"Invalid --months value: {args.months!r} (expected positive integer).")
            return 2
    width_v = exact_pos_int(args.width, default=None)
    height_v = exact_pos_int(args.height, default=None)
    if width_v is None or height_v is None:
        print(f"Invalid --width/--height values: width={args.width!r}, height={args.height!r}.")
        return 2

    path = str(resolve_cachedb_filepath(os.path.join(args.cache_dir, CACHE_LONG_HORIZON_FILENAME)))
    data = read_json_dict(path, default={})
    items = data.get("items") if isinstance(data, dict) and isinstance(data.get("items"), dict) else {}

    if not args.symbol:
        if not items:
            print(f"No long-horizon snapshot found at {path}.")
            return 1
        print(f"Available symbols ({data.get('horizon_label', '?')} / {data.get('months', '?')}m): {', '.join(sorted(items.keys()))}")
        return 0

    req_raw = args.symbol.strip()
    req_u = req_raw.upper()
    matched_sym = req_raw
    matched = items.get(req_raw)
    if matched is None:
        for k, v in items.items():
            if k.upper() == req_u:
                matched_sym = k
                matched = v
                break

    archived_daily = load_ohlc_archive_bars(args.cache_dir, matched_sym, timeframe="1d")
    intraday_bars = None
    if args.tf in ("4h", "1h"):
        intraday_bars = load_ohlc_archive_bars(args.cache_dir, matched_sym, timeframe=args.tf)
        if not intraday_bars and args.tf == "4h":
            bars_1h = load_ohlc_archive_bars(args.cache_dir, matched_sym, timeframe="1h")
            if bars_1h:
                intraday_bars = aggregate_ohlc_bars(bars_1h, target_interval_min=240)

    if not isinstance(matched, dict):
        if archived_daily:
            matched = evaluate_symbol_long_horizon(
                matched_sym,
                archived_daily,
                months=months_v,
                include_bars=False,
            )
        elif intraday_bars:
            matched = {"symbol": matched_sym}
        if not isinstance(matched, dict):
            avail = ", ".join(sorted(items.keys())) if items else "none"
            print(f"Symbol {args.symbol!r} not found. Available: {avail}")
            return 1

    print(
        render_ascii_chart_view(
            matched,
            daily_bars=archived_daily or None,
            bars_override=intraday_bars or None,
            start_ym=args.start_ym,
            end_ym=args.end_ym,
            months=months_v,
            timeframe=args.tf,
            width=width_v,
            height=height_v,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
