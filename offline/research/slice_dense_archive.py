#!/usr/bin/env python3
"""slice_dense_archive.py -- extract the last N days from a dense archive
(cachedb/cache_365_price_{symbol}.jsonl) into a compact file for fast sweeps
on parameters with short time constants (fire-retry/cooldown), where long
sparse history produces sampling artifacts (see
offline/research/tradeall_trigger_gate/README.md, Exp 6/7) and the full dense
archive has grown too large for iterative sweeps (~1M points/symbol).

Usage: slice_dense_archive.py <symbol> <days> [out_path]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import normalize_epoch_sec, safe_pos_float
from cachepaths import cache_365_price_filename, cache_path, cache_symbol_variants
from state_io import atomic_write_jsonl, iter_jsonl_dicts
from strategies.replay_metrics import validate_bt_cli_token


def ts_of(r) -> float:
    if not isinstance(r, dict):
        return 0.0
    i_val = r.get("i")
    raw = i_val[0] if isinstance(i_val, (list, tuple)) and i_val else None
    if raw is None:
        raw = r.get("t")
    return safe_pos_float(raw, 0.0)


def _ts_sec(r) -> float:
    raw_ts = ts_of(r)
    if raw_ts <= 0.0:
        return 0.0
    sec = normalize_epoch_sec(raw_ts, default=0.0)
    return sec if sec is not None and sec > 0.0 else 0.0


def main(argv: list[str] | None = None):
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) < 2:
        print("Usage: slice_dense_archive.py <symbol> <days> [out_path]", file=sys.stderr)
        sys.exit(1)
    try:
        symbol = validate_bt_cli_token(args[0], "symbol").upper()
    except ValueError:
        print(f"[slice] invalid symbol={args[0]!r}", file=sys.stderr)
        sys.exit(1)
    days = safe_pos_float(args[1], default=None)
    if days is None:
        print(f"[slice] invalid days={args[1]!r}", file=sys.stderr)
        sys.exit(1)
    out = args[2] if len(args) > 2 else cache_path(
        f"cache_24_price_slice_{symbol}_{days:g}d.jsonl")
    src = next(
        (
            p
            for cand in cache_symbol_variants(symbol)
            if os.path.isfile(p := cache_path(cache_365_price_filename(cand)))
        ),
        cache_path(cache_365_price_filename(symbol)),
    )
    if not os.path.isfile(src):
        print(f"[slice] {src} missing", file=sys.stderr)
        sys.exit(1)

    total_rows = 0
    last_ts = 0.0
    for rec in iter_jsonl_dicts(src):
        t = _ts_sec(rec)
        if t > 0.0:
            total_rows += 1
            if t > last_ts:
                last_ts = t
    if total_rows == 0:
        print(f"[slice] {src} empty/missing", file=sys.stderr)
        sys.exit(1)

    cutoff = last_ts - days * 86400.0
    sliced: list[dict] = []
    prev_t, is_sorted = -1.0, True
    for rec in iter_jsonl_dicts(src):
        t = _ts_sec(rec)
        if t >= cutoff and t > 0.0:
            if rec.get("s") != symbol:
                rec = {**rec, "s": symbol}
            if t < prev_t:
                is_sorted = False
            prev_t = t
            sliced.append(rec)
    if not is_sorted:
        sliced.sort(key=_ts_sec)

    atomic_write_jsonl(out, sliced)

    print(f"[slice] {symbol}: {total_rows} -> {len(sliced)} points (last {days}d) -> {out}")


if __name__ == "__main__":
    main()
