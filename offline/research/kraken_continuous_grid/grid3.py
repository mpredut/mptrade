#!/usr/bin/env python3
"""Wave 3: continuous full-history runs with a FINITE cash account of 3900 USD (replay
initial_cash): losses are not refilled, open BUYs reserve cash. Top candidates only."""
import json, os, sys, traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import int_env
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
        cache_key = (asset, len(bars), bars[0][0] if bars else 0, bars[-1][0] if bars else 0,
                     repr(params), bool(peak))
        cached = _TASK3_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cfg": cfg}
        out = {
            str(fee): grid.replay(ohlc[WARM:], ohlc[:WARM], params, peak, fee=fee, initial_cash=3900.0)
            for fee in (0.26, 0.40)
        }
        res = {"asset": asset, "cfg": cfg, **out}
        _TASK3_CACHE[cache_key] = res
        return res
    except Exception:
        return {"asset": asset, "cfg": cfg, "error": traceback.format_exc()}


if __name__ == "__main__":
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    for a in assets:
        grid.load_asset(a)
    todo = [(a, c) for a in assets for c in CFGS]
    canon_groups: dict = {}
    canon_todo = []
    for a, c in todo:
        p, pk = grid2.ALL[c]
        k = (a, repr(p), bool(pk))
        if k not in canon_groups:
            canon_groups[k] = []
            canon_todo.append((a, c))
        canon_groups[k].append(c)
    done_count = 0
    workers = int_env("WORKERS") or (os.cpu_count() or 4)
    with Pool(workers) as pool, open(os.path.join(EXP, "results3.jsonl"), "w") as fh:
        for r in pool.imap_unordered(task, canon_todo):
            cfg_entry = grid2.ALL.get(r.get("cfg"))
            aliases = (
                canon_groups.get((r["asset"], repr(cfg_entry[0]), bool(cfg_entry[1])), [r["cfg"]])
                if cfg_entry is not None
                else [r.get("cfg", "?")]
            )
            for idx, alias_cfg in enumerate(aliases):
                done_count += 1
                row = r if idx == 0 and alias_cfg == r["cfg"] else {**r, "cfg": alias_cfg}
                fh.write(json.dumps(row) + "\n")
                if "error" in row:
                    print(f"[grid3] {done_count}/{len(todo)} {row['asset']:<5} {row['cfg']:<24} ERROR", flush=True)
                else:
                    print(f"[grid3] {done_count}/{len(todo)} {row['asset']:<5} {row['cfg']:<24} {row.get('0.26', {})} ", flush=True)
            fh.flush()
