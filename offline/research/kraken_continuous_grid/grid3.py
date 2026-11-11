#!/usr/bin/env python3
"""Wave 3: continuous full-history runs with a FINITE cash account of 3900 USD (replay
initial_cash): losses are not refilled, open BUYs reserve cash. Top candidates only."""
import os, sys, traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import lru_dict_put
from state_io import append_jsonl_record, atomic_write_jsonl
from offline.research.kraken_continuous_grid import grid, grid2
load, WARM, DATA, EXP = grid.load, grid.WARM, grid.DATA, grid.EXP
CFGS = ["live", "old_live_pre0923", "sl0", "abl_gate_off", "ov650_t8", "ov650_t8_brake",
        "ov650_t8_brake_gateoff", "ov650_t8_gateoff", "ov650_t8_sl0", "ov650_t8_sl25",
        "ov1000_t8_gateoff", "ov1000_t10_gateoff", "ov2000_t10", "ov1300_t12", "ov800_t12"]

_TASK3_CACHE: dict = {}


def task(args):
    asset, cfg = args
    try:
        bars, ohlc, _ = grid.load_asset(asset)
        params, peak = grid2.ALL[cfg]
        start_ts, end_ts = grid._bar_ts_bounds(bars)
        cache_key = (os.path.abspath(grid.DATA), asset, len(bars), start_ts, end_ts,
                     repr(params), bool(peak))
        cached = _TASK3_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cfg": cfg}
        out = {
            str(fee): grid.replay(bars[WARM:], bars[:WARM], params, peak, fee=fee, initial_cash=3900.0)
            for fee in (0.26, 0.40)
        }
        res = {
            "asset": asset,
            "cfg": cfg,
            "bars": len(bars),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "params_repr": repr(params),
            "peak": bool(peak),
            **out,
        }
        lru_dict_put(_TASK3_CACHE, cache_key, res, max_entries=512)
        return res
    except Exception:
        return {"asset": asset, "cfg": cfg, "error": traceback.format_exc()}


if __name__ == "__main__":
    if not os.path.isdir(DATA):
        print(f"[grid3] ERROR: data directory missing: {DATA}", file=sys.stderr, flush=True)
        sys.exit(1)
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    if not assets:
        print(f"[grid3] ERROR: no *_240m.csv assets found in {DATA}", file=sys.stderr, flush=True)
        sys.exit(1)
    for a in assets:
        grid.load_asset(a)
    todo = [(a, c) for a in assets for c in CFGS]
    canon_groups: dict = {}
    canon_aliases: dict[tuple[str, str], list[str]] = {}
    canon_todo = []
    for a, c in todo:
        spec = grid2.ALL.get(c)
        if spec is not None:
            p, pk = spec
            k = (a, repr(p), bool(pk))
        else:
            k = (a, c, False)
        if k not in canon_groups:
            aliases_list: list[str] = []
            canon_groups[k] = aliases_list
            canon_aliases[(a, c)] = aliases_list
            canon_todo.append((a, c))
        canon_groups[k].append(c)
    done_count = 0
    valid_rows = 0
    workers = grid.resolve_grid_workers()
    out3_path = os.path.join(EXP, "results3.jsonl")
    atomic_write_jsonl(out3_path, ())
    with Pool(workers) as pool:
        for r in pool.imap_unordered(task, canon_todo):
            r_asset = str(r.get("asset", "?"))
            r_cfg = str(r.get("cfg", "?"))
            aliases = canon_aliases.get((r_asset, r_cfg), [r_cfg])
            for idx, alias_cfg in enumerate(aliases):
                done_count += 1
                row = r if idx == 0 and alias_cfg == r_cfg else {**r, "cfg": alias_cfg}
                append_jsonl_record(out3_path, row)
                if "error" in row:
                    print(f"[grid3] {done_count}/{len(todo)} {row['asset']:<5} {row['cfg']:<24} ERROR", flush=True)
                else:
                    valid_rows += 1
                    print(f"[grid3] {done_count}/{len(todo)} {row['asset']:<5} {row['cfg']:<24} {row.get('0.26', {})} ", flush=True)
    if valid_rows == 0:
        print("[grid3] ERROR: zero valid rows produced", file=sys.stderr, flush=True)
        sys.exit(1)
