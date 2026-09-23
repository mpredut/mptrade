#!/usr/bin/env python3
"""Wave 3: continuous full-history runs with a FINITE cash account of 3900 USD (replay
initial_cash): losses are not refilled, open BUYs reserve cash. Top candidates only."""
import contextlib, io, json, os, sys, time
from multiprocessing import Pool
import grid, grid2
from grid import load, WARM, DATA, EXP
CFGS = ["live", "old_live_pre0923", "sl0", "abl_gate_off", "ov650_t8", "ov650_t8_brake",
        "ov650_t8_brake_gateoff", "ov650_t8_gateoff", "ov650_t8_sl0", "ov650_t8_sl25",
        "ov1000_t8_gateoff", "ov1000_t10_gateoff", "ov2000_t10", "ov1300_t12", "ov800_t12"]

def task(args):
    asset, cfg = args
    bars = load(os.path.join(DATA, f"{asset}_240m.csv"))
    params, peak = grid2.ALL[cfg]
    grid.FLAGS["peak"] = bool(peak)
    out = {}
    for fee in (0.26, 0.40):
        with contextlib.redirect_stdout(io.StringIO()):
            m = grid.rp.run_replay([b[1:] for b in bars[WARM:]], params, fee_pct=fee, bar_minutes=240,
                                   warmup_ohlc=[b[1:] for b in bars[:WARM]], initial_cash=3900.0)
        out[str(fee)] = {"ret": round(m["total"] / 3900 * 100, 2), "dd": round(m.get("max_drawdown_pct") or 0, 2),
                         "cyc": m.get("cycles", 0), "refused": m["funding"]["refused_buys"],
                         "min_cash": m["funding"]["minimum_cash"]}
    return {"asset": asset, "cfg": cfg, **out}

if __name__ == "__main__":
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    todo = [(a, c) for c in CFGS for a in assets]
    with Pool(4) as pool, open(os.path.join(EXP, "results3.jsonl"), "w") as fh:
        for i, r in enumerate(pool.imap_unordered(task, todo), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"[grid3] {i}/{len(todo)} {r['asset']:<5} {r['cfg']:<24} {r['0.26']} ", flush=True)
