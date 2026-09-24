#!/usr/bin/env python3
"""Wave 2 (DEV): walk-forward config selection, fine overlay grid, fee stress.

Reuses grid.py (same live engine, same TTL harness correction, same datasets).

  select  - every 90 days, rank a candidate set by its trailing-365-day continuous
            return/drawdown and run the winner on the NEXT 90 days (fresh state, 40-bar
            warm-up). Compared with every static candidate on the same out-of-sample
            segments: does choosing parameters from backtests beat simply keeping one?
  fine    - overlay top-up/trail grid around the wave-1 region plus combinations with
            the individually reverted 23 Sep changes; same views as wave 1.
  fee     - key configs at 0.40%/leg instead of 0.26% (execution-cost stress).
"""
import contextlib, io, json, os, statistics as st, sys, time, traceback
from multiprocessing import Pool

import grid
from grid import CONFIGS, LIVE, OLD, R, ov, load, WARM, W30, W90, WSTEP, DATA, EXP

OUT = os.path.join(EXP, "results2.jsonl")
FEE = {"std": 0.26, "stress": 0.40}

FINE = {
    **{f"ov{t}_t{tr}": (ov(LIVE, t, tr), 0) for t in (500, 800, 1300) for tr in (7, 9, 12)},
    "ov650_t8_gateoff": (ov(R(LIVE, tp_regime_gate=False), 650, 8), 0),
    "ov650_t8_tp5": (ov(R(LIVE, takeprofit_pct=5.0), 650, 8), 0),
    "ov650_t8_sl25": (ov(R(LIVE, stop_loss_pct=25.0), 650, 8), 0),
    "ov650_t8_sl0": (ov(R(LIVE, stop_loss_pct=0.0), 650, 8), 0),
    "ov650_t8_reentry4": (ov(R(LIVE, reentry_drop_pct=4.0), 650, 8), 0),
    "ov650_t8_dca1.5": (ov(R(LIVE, dca_drop_pct=1.5), 650, 8), 0),
    "ov1000_t10_gateoff": (ov(R(LIVE, tp_regime_gate=False), 1000, 10), 0),
    "ov1000_t8_gateoff_tp5": (ov(R(LIVE, tp_regime_gate=False, takeprofit_pct=5.0), 1000, 8), 0),
    "gateoff_tp5": (R(LIVE, tp_regime_gate=False, takeprofit_pct=5.0), 0),
    "gateoff_trail5": (R(LIVE, tp_regime_gate=False, tp_trail_adaptive=False, tp_trail_pct=5.0), 0),
    "ov650_t8_brake_gateoff": (R(ov(R(LIVE, tp_regime_gate=False), 650, 8),
                                 dca_trend_brake=True, dca_brake_min_pct=1.5), 0),
}
ALL = {**CONFIGS, **FINE}
SELECT_SET = ["live", "old_live_pre0923", "abl_gate_off", "abl_tp5", "tp_classic", "peak2.2",
              "ov350_t6", "ov650_t8", "ov1000_t8", "ov650_t10", "ov2000_t8",
              "ov650_t8_gateoff", "ov1000_t10_gateoff", "ov650_t8_sl12.5"]
FEE_SET = ["live", "old_live_pre0923", "abl_gate_off", "ov350_t6", "ov650_t8", "ov1000_t8",
           "ov650_t8_gateoff", "tp_classic"]
TRAIN, TEST = 2190, 540          # 365 days and 90 days of 4h bars


def replay(bars, warm, params, peak, fee):
    grid.FLAGS["peak"] = bool(peak)
    with contextlib.redirect_stdout(io.StringIO()):
        m = grid.rp.run_replay([b[1:] for b in bars], params, fee_pct=fee, bar_minutes=240,
                               warmup_ohlc=[b[1:] for b in warm])
    budget = float(params.effective_max_budget())
    return {"ret": round(m["total"] / budget * 100, 3), "dd": round(m.get("max_drawdown_pct") or 0, 3),
            "cyc": m.get("cycles", 0), "expo": round(m.get("exposure_pct") or 0, 1),
            "bh": round((bars[-1][4] / bars[0][4] - 1) * 100, 3)}


def views(asset, cfg, fee_key):
    bars = load(os.path.join(DATA, f"{asset}_240m.csv"))
    params, peak = ALL[cfg]
    fee = FEE[fee_key]
    res = {"kind": "views", "fee": fee_key, "asset": asset, "cfg": cfg, "bars": len(bars),
           "full": replay(bars[WARM:], bars[:WARM], params, peak, fee), "w30": [], "w90": []}
    for key, size in (("w30", W30), ("w90", W90)):
        for s in range(WARM, len(bars) - size + 1, WSTEP):
            r = replay(bars[s:s + size], bars[s - WARM:s], params, peak, fee)
            res[key].append([r["ret"], r["dd"], r["bh"], r["cyc"]])
    return res


def select(asset):
    bars = load(os.path.join(DATA, f"{asset}_240m.csv"))
    steps = []
    for t in range(WARM + TRAIN, len(bars) - TEST + 1, TEST):
        train = {c: replay(bars[t - TRAIN:t], bars[t - TRAIN - WARM:t - TRAIN], *ALL[c], 0.26)
                 for c in SELECT_SET}
        score = {c: r["ret"] / max(r["dd"], 5.0) for c, r in train.items()}
        best = max(score, key=score.get)
        oos = {c: replay(bars[t:t + TEST], bars[t - WARM:t], *ALL[c], 0.26)["ret"] for c in SELECT_SET}
        steps.append({"t": bars[t][0], "pick": best, "pick_ret": oos[best], "oos": oos,
                      "bh": round((bars[t + TEST - 1][4] / bars[t][4] - 1) * 100, 2)})
    return {"kind": "select", "asset": asset, "steps": steps}


def task(args):
    try:
        if args[0] == "select":
            t0 = time.time(); r = select(args[1]); r["secs"] = round(time.time() - t0, 1); return r
        t0 = time.time(); r = views(*args[1:]); r["secs"] = round(time.time() - t0, 1); return r
    except Exception:
        return {"kind": "error", "args": list(args), "error": traceback.format_exc()}


if __name__ == "__main__":
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    todo = [("select", a) for a in assets]
    todo += [("views", a, c, "std") for c in FINE for a in assets]
    todo += [("views", a, c, "stress") for c in FEE_SET for a in assets]
    print(f"[grid2] {len(todo)} tasks", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", 3))) as pool, open(OUT, "w") as out:
        for i, r in enumerate(pool.imap_unordered(task, todo), 1):
            out.write(json.dumps(r) + "\n"); out.flush()
            tag = r.get("kind")
            if tag == "views":
                tag = f"{r['asset']:<5} {r['cfg']:<24} {r['fee']:<6} full {r['full']['ret']:+8.2f}% dd {r['full']['dd']:6.2f}"
            elif tag == "select":
                tag = f"select {r['asset']} ({len(r['steps'])} steps)"
            print(f"[grid2] {i}/{len(todo)} {tag} [{time.time()-t0:.0f}s]", flush=True)
    print("[grid2] done", flush=True)
