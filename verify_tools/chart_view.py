#!/usr/bin/env python3
"""Read-only on-demand long-horizon chart zoom utility for AI executors and operators.

Reads ``cachedb/macro/long_horizon_collect.json`` (no network I/O) and renders an exact ASCII
price chart with OHLCV anchors and ZigZag swings for any chosen period (``--months N`` or
``--from YYYY-MM --to YYYY-MM``).
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
from intelligence.sentiment.evaluators.long_horizon_evaluator import render_ascii_chart_view
from state_io import read_json_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect long-horizon OHLCV chart slice from cachedb/macro/long_horizon_collect.json")
    parser.add_argument("symbol", nargs="?", help="Symbol to inspect (e.g. BTCUSDC, HYPEUSDC, NVDA_US_EQ); omit to list available symbols")
    parser.add_argument("--months", default=None, help="Trailing number of months to zoom into (e.g. 6, 12, 18, 24, 36)")
    parser.add_argument("--from", dest="start_ym", default=None, help="Start date YYYY-MM or YYYY-MM-DD (UTC)")
    parser.add_argument("--to", dest="end_ym", default=None, help="End date YYYY-MM or YYYY-MM-DD (UTC)")
    parser.add_argument("--tf", choices=("1w", "1d"), default="1w", help="Timeframe: 1w (full multi-year window) or 1d (recent 90d)")
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
    data = read_json_dict(path)
    items = data.get("items") if isinstance(data, dict) and isinstance(data.get("items"), dict) else {}
    if not items:
        print(f"No long-horizon snapshot found at {path}.")
        return 1

    if not args.symbol:
        print(f"Available symbols ({data.get('horizon_label', '?')} / {data.get('months', '?')}m): {', '.join(sorted(items.keys()))}")
        return 0

    req_u = args.symbol.strip().upper()
    matched = items.get(args.symbol.strip())
    if matched is None:
        for k, v in items.items():
            if k.upper() == req_u:
                matched = v
                break
    if not isinstance(matched, dict):
        print(f"Symbol {args.symbol!r} not found. Available: {', '.join(sorted(items.keys()))}")
        return 1

    print(
        render_ascii_chart_view(
            matched,
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
