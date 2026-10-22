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
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    symbol = sys.argv[1]
    days = float(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        ROOT, "cachedb", f"cache_24_price_slice_{symbol}_{days:g}d.jsonl")
    src = os.path.join(ROOT, "cachedb", f"cache_365_price_{symbol}.jsonl")
    if not os.path.exists(src):
        src = os.path.join(ROOT, "cachedb", f"cache_365price_{symbol}.jsonl")

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print(f"[slice] {src} gol/missing", file=sys.stderr)
        sys.exit(1)

    def ts_of(r):
        return r.get("i", [None])[0] or r.get("t")

    rows.sort(key=ts_of)
    cutoff = ts_of(rows[-1]) - days * 86400 * 1000
    sliced = [r for r in rows if ts_of(r) >= cutoff]

    with open(out, "w", encoding="utf-8") as f:
        for r in sliced:
            f.write(json.dumps(r) + "\n")

    print(f"[slice] {symbol}: {len(rows)} -> {len(sliced)} puncte (ultimele {days}z) -> {out}")


if __name__ == "__main__":
    main()
