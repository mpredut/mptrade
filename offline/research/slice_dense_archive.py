#!/usr/bin/env python3
"""slice_dense_archive.py — extrage ultimele N zile dintr-o arhiva DENSA
(cachedb/cache_365_price_{symbol}.jsonl) into a small file, for FAST sweeps
on parameters with SHORT time constants (fire-retry/cooldown), where
istoricul lung/sparse ar produce artefacte (vezi offline/research/tradeall_trigger_gate/
README.md, Exp 6/7) AND the complete dense archive has grown too large to be
rulata intreaga (14 iul->azi: ~1M puncte/simbol, ~8h/rulare la rata observata).

Usage: slice_dense_archive.py <symbol> <days> [out_path]
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ts_of(r) -> float:
    if not isinstance(r, dict):
        return 0.0
    i_val = r.get("i")
    raw = i_val[0] if isinstance(i_val, list) and i_val else None
    if raw is None:
        raw = r.get("t")
    if isinstance(raw, bool) or raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return val if math.isfinite(val) and val > 0 else 0.0


def main():
    if len(sys.argv) < 3:
        print("Usage: slice_dense_archive.py <symbol> <days> [out_path]", file=sys.stderr)
        sys.exit(1)
    symbol = sys.argv[1]
    try:
        days = float(sys.argv[2])
    except (TypeError, ValueError, OverflowError):
        days = -1.0
    if not (math.isfinite(days) and days > 0):
        print(f"[slice] invalid days={sys.argv[2]!r}", file=sys.stderr)
        sys.exit(1)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        ROOT, "cachedb", f"cache_24_price_slice_{symbol}_{days:g}d.jsonl")
    src = os.path.join(ROOT, "cachedb", f"cache_365_price_{symbol}.jsonl")

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ts_of(rec) > 0:
                rows.append(rec)
    if not rows:
        print(f"[slice] {src} gol/missing", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=ts_of)
    last_ts = ts_of(rows[-1])
    ts_scale = 1000.0 if last_ts > 1e11 else 1.0
    cutoff = last_ts - days * 86400.0 * ts_scale
    sliced = [r for r in rows if ts_of(r) >= cutoff]

    with open(out, "w", encoding="utf-8") as f:
        for r in sliced:
            f.write(json.dumps(r) + "\n")

    print(f"[slice] {symbol}: {len(rows)} -> {len(sliced)} puncte (ultimele {days}z) -> {out}")


if __name__ == "__main__":
    main()
