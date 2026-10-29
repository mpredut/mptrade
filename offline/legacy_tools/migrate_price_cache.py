#!/usr/bin/env python3
"""
Migrates cache_price_<SYM>.json (old, full rewrite) -> cache_price_<SYM>.jsonl (append).

During the refactor we moved from .json to .jsonl plus .meta, but the old data in .json
was never migrated. This script:
  • reads the points from the old .json and from the existing .jsonl (if there is one),
  • le imbina per simbol, deduplica pe timestamp, sorteaza crescator,
  • rescrie .jsonl atomic + regenereaza .meta,
  • renames the old .json -> .json.bak (you can delete the .bak after checking).

RUN IT ONLY with the writer processes (priceAnalysis and so on) STOPPED, so there is no
concurrent write on the .jsonl. Idempotent (re-running does not duplicate).
"""
import os
import sys
import json
import glob
import time

# offline/legacy_tools/ sits two levels below the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from botcore import safe_pos_float
from cachepaths import get_cachedb_dir
from state_io import atomic_write_json, atomic_write_text, read_json_dict, read_jsonl_dicts

CACHE_DIR = get_cachedb_dir()


def _entry_ts(item):
    # An item is [ts, price] (a list) — we use the first element as the timestamp.
    raw = 0
    if isinstance(item, (list, tuple)) and item:
        raw = item[0]
    elif isinstance(item, dict):
        raw = item.get("time") if item.get("time") is not None else item.get("timestamp")
    val = safe_pos_float(raw, default=None)
    return int(val) if val is not None else 0


def _load_old_json(path):
    """{'items': {SYM: [[ts,price],...]}, 'fetchtime': {...}} → {SYM: [items]}"""
    data = read_json_dict(path, default={})
    items = data.get("items", data) if isinstance(data, dict) else {}
    if not isinstance(items, dict):
        return {}
    return {k: v for k, v in items.items() if isinstance(v, list)}


def _load_jsonl(path):
    """{SYM: [items]} from the lines {'s': SYM, 'i': item}."""
    out = {}
    for rec in read_jsonl_dicts(path):
        if "s" in rec and "i" in rec:
            out.setdefault(rec["s"], []).append(rec["i"])
    return out


def migrate_file(json_path, dry_run=False):
    sym_from_name = os.path.basename(json_path)[len("cache_price_"):-len(".json")]
    jsonl_path = os.path.join(os.path.dirname(json_path), f"cache_sparse_price_{sym_from_name}.jsonl")

    old = _load_old_json(json_path)
    new = _load_jsonl(jsonl_path)

    symbols = set(old) | set(new)
    merged = {}
    added_total = 0
    for sym in symbols:
        by_ts = {}
        for item in new.get(sym, []):          # intai noile (au prioritate la duplicat)
            ts = _entry_ts(item)
            if ts > 0:
                by_ts[ts] = item
        before = len(by_ts)
        for item in old.get(sym, []):
            ts = _entry_ts(item)
            if ts > 0:
                by_ts.setdefault(ts, item)
        added = len(by_ts) - before
        added_total += added
        merged[sym] = [by_ts[k] for k in sorted(by_ts)]

    # scriere
    lines, max_ts, counts, fetchtime = [], 0, {}, {}
    for sym, items in merged.items():
        counts[sym] = len(items)
        for item in items:
            lines.append(json.dumps({"s": sym, "i": item}))
            max_ts = max(max_ts, _entry_ts(item))
        if items:
            fetchtime[sym] = _entry_ts(items[-1])

    print(f"[{sym_from_name}] old={sum(len(v) for v in old.values())} "
          f"jsonl={sum(len(v) for v in new.values())} → merged={sum(counts.values())} "
          f"(+{added_total} from the old one)")

    if dry_run:
        return added_total

    atomic_write_text(jsonl_path, "".join(ln + "\n" for ln in lines))
    atomic_write_json(jsonl_path + ".meta", {"max_ts": max_ts, "saved_at": int(time.time() * 1000),
                                            "fetchtime": fetchtime, "counts": counts})
    os.replace(json_path, json_path + ".bak")     # A backup; we do not delete directly.
    return added_total


def _arg_value(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    dry_run = "--dry-run" in sys.argv
    work_dir = _arg_value("--dir", CACHE_DIR)   # cachedb/ by default (or --dir / BINANCE_CACHE_DIR)
    # cache_price_long_trend.json is the PriceLongTrend cache (a full-rewrite json, NOT
    # a per-symbol append) -> it stays .json, and we exclude it from the migration.
    EXCLUDE = {"long_trend"}
    files = sorted(glob.glob(os.path.join(glob.escape(work_dir), "cache_price_*.json")))
    files = [f for f in files
             if not f.endswith(".bak")
             and os.path.basename(f)[len("cache_price_"):-len(".json")] not in EXCLUDE]
    if not files:
        print("There are no cache_price_*.json files to migrate.")
        return
    print(f"{'DRY RUN' if dry_run else 'MIGRATION'}: {len(files)} files\n")
    for f in files:
        try:
            migrate_file(f, dry_run=dry_run)
        except Exception as e:
            print(f"[EROARE] {os.path.basename(f)}: {e}")
    if not dry_run:
        print("\nGata. Fisierele .json au fost redenumite in .json.bak "
              "(delete them once you have checked the .jsonl).")


if __name__ == "__main__":
    main()
