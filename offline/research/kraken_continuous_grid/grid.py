#!/usr/bin/env python3
"""Kraken spot-DCA research grid over the widest available 4h history (runs on DEV).

Every configuration is replayed through the live engine (kraken/replay.py ->
strategies/spot_dca.Strategy) on 4h bars, three ways:
  full    - one continuous run over the whole dataset: state carries across cycles,
            which is what the live bot actually experiences (incl. the post-TP re-entry lock);
  year    - continuous runs per calendar year (regime robustness);
  w30/w90 - rolling fresh-state windows of 30 and 90 days, step 30 days (what the
            existing walk-forward/pilot tooling measures).

The replay engine re-places a stale limit BUY like live reconcile() does after the TTL.
One harness-only variant (FLAGS["peak"] in _step): after a TP exit, ratchet last_sell_price
to the highest close since the exit, i.e. re-enter on a pullback from the post-exit peak.

Data comes from fetch_data.py; results go to $GRID_OUT (default
offline/results/kraken_continuous_grid). Run everything with run_all.sh.
"""
import contextlib, csv, dataclasses, io, json, os, sys, time, traceback
from datetime import datetime, timezone
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
DATA = os.path.join(EXP, "data")
OUT = os.path.join(EXP, "results.jsonl")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kraken"))
os.chdir(ROOT)
from kraken import shadow_live  # noqa: E402
shadow_live._load_runtime_config()
from strategies import spot_dca as strat  # noqa: E402
import replay as rp  # noqa: E402

FLAGS = {"peak": False}
_orig_step = strat.Strategy.step


def _step(self, price, timestamp=None):
    s = self.s
    if s["qty"] <= 1e-12:
        if FLAGS["peak"] and s.get("last_exit_kind") == "TP" and s.get("last_sell_price"):
            s["last_sell_price"] = max(s["last_sell_price"], price)
    return _orig_step(self, price, timestamp=timestamp)


strat.Strategy.step = _step
strat.log = lambda *a, **k: None

R = dataclasses.replace
LIVE = strat.StratParams.from_env()
OLD = R(LIVE, takeprofit_pct=5.0, dca_spacing_growth_pct=0.0, tp_trail_adaptive=False,
        tp_regime_gate=False, tp_trail_profit_floor_pct=0.0, stop_loss_pct=12.5)


def ov(p, topup, trail, brk=False):
    return R(p, trend_overlay=True, trend_topup=float(topup), trend_trail_pct=float(trail),
             trend_exit_break=brk)


CONFIGS = {
    # A. baselines
    "live": (LIVE, 0), "old_live_pre0923": (OLD, 0),
    # B. ablation of the 23 Sep promotion (live with ONE change reverted)
    "abl_tp5": (R(LIVE, takeprofit_pct=5.0), 0),
    "abl_spacing0": (R(LIVE, dca_spacing_growth_pct=0.0), 0),
    "abl_trail_fixed": (R(LIVE, tp_trail_adaptive=False), 0),
    "abl_gate_off": (R(LIVE, tp_regime_gate=False), 0),
    "abl_floor0": (R(LIVE, tp_trail_profit_floor_pct=0.0), 0),
    "abl_sl12.5": (R(LIVE, stop_loss_pct=12.5), 0),
    # C. trend overlay grid on live
    **{f"ov{t}_t{tr}": (ov(LIVE, t, tr), 0) for t in (350, 650, 1000, 2000) for tr in (4, 6, 8, 10)},
    "ov650_t8_break": (ov(LIVE, 650, 8, True), 0),
    "ov650_t8_oldlive": (ov(OLD, 650, 8), 0),
    "ov650_t8_sl12.5": (ov(R(LIVE, stop_loss_pct=12.5), 650, 8), 0),
    "ov1000_t8_gateoff": (ov(R(LIVE, tp_regime_gate=False), 1000, 8), 0),
    "ov650_t8_brake": (R(ov(LIVE, 650, 8), dca_trend_brake=True, dca_brake_min_pct=1.5), 0),
    # D. re-entry
    "reentry0": (R(LIVE, reentry_drop_pct=0.0), 0),
    "reentry1": (R(LIVE, reentry_drop_pct=1.0), 0),
    "reentry4": (R(LIVE, reentry_drop_pct=4.0), 0),
    "reentry_adaptive": (R(LIVE, reentry_adaptive=True), 0),
    "peak2.2": (LIVE, 1), "peak4": (R(LIVE, reentry_drop_pct=4.0), 1),
    "peak2.2_oldlive": (OLD, 1),
    "ov650_t8_peak4": (ov(R(LIVE, reentry_drop_pct=4.0), 650, 8), 1),
    # E. stop loss
    "sl0": (R(LIVE, stop_loss_pct=0.0), 0), "sl8": (R(LIVE, stop_loss_pct=8.0), 0),
    "sl25": (R(LIVE, stop_loss_pct=25.0), 0),
    # F. take profit
    "tp3": (R(LIVE, takeprofit_pct=3.0), 0), "tp6": (R(LIVE, takeprofit_pct=6.0), 0),
    "tp8": (R(LIVE, takeprofit_pct=8.0), 0),
    "tp_classic": (R(LIVE, tp_trend_hold=False), 0),
    # G. DCA spacing
    "dca0.75": (R(LIVE, dca_drop_pct=0.75), 0), "dca1.5": (R(LIVE, dca_drop_pct=1.5), 0),
    "dca2.5": (R(LIVE, dca_drop_pct=2.5), 0),
    "spacing0.5": (R(LIVE, dca_spacing_growth_pct=0.5), 0),
    # I. risk overlays
    "dca_brake": (R(LIVE, dca_trend_brake=True, dca_brake_min_pct=1.5), 0),
    "dca_vol_m1": (R(LIVE, dca_vol_scale_k=-1.0, dca_vol_ref=2.0), 0),
    "dca_vol_p1": (R(LIVE, dca_vol_scale_k=1.0, dca_vol_ref=2.0), 0),
}

W30, W90, WSTEP, WARM = 180, 540, 180, 40


def load(path):
    with open(path) as fh:
        return [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"])) for r in csv.DictReader(fh)]


def replay(bars, warm, params, peak):
    FLAGS["peak"] = bool(peak)
    with contextlib.redirect_stdout(io.StringIO()):
        m = rp.run_replay([b[1:] for b in bars], params, fee_pct=0.26, bar_minutes=240,
                          warmup_ohlc=[b[1:] for b in warm])
    budget = float(params.effective_max_budget())
    return {"ret": round(m["total"] / budget * 100, 3), "dd": round(m.get("max_drawdown_pct") or 0, 3),
            "cyc": m.get("cycles", 0), "expo": round(m.get("exposure_pct") or 0, 1),
            "bh": round((bars[-1][4] / bars[0][4] - 1) * 100, 3)}


def task(args):
    asset, cfg = args
    try:
        bars = load(os.path.join(DATA, f"{asset}_240m.csv"))
        params, peak = CONFIGS[cfg]
        t0 = time.time()
        res = {"asset": asset, "cfg": cfg, "bars": len(bars),
               "full": replay(bars[WARM:], bars[:WARM], params, peak), "year": {}, "w30": [], "w90": []}
        years = sorted({datetime.fromtimestamp(b[0], timezone.utc).year for b in bars})
        for y in years:
            idx = [i for i, b in enumerate(bars) if datetime.fromtimestamp(b[0], timezone.utc).year == y]
            if len(idx) < 360 or idx[0] < WARM:
                continue
            res["year"][str(y)] = replay(bars[idx[0]:idx[-1] + 1], bars[idx[0] - WARM:idx[0]], params, peak)
        for key, size in (("w30", W30), ("w90", W90)):
            for s in range(WARM, len(bars) - size + 1, WSTEP):
                r = replay(bars[s:s + size], bars[s - WARM:s], params, peak)
                res[key].append([r["ret"], r["dd"], r["bh"], r["cyc"]])
        res["secs"] = round(time.time() - t0, 1)
        return res
    except Exception:
        return {"asset": asset, "cfg": cfg, "error": traceback.format_exc()}


if __name__ == "__main__":
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["asset"], r["cfg"]))
    first = ["live", "old_live_pre0923", "ov650_t8", "ov350_t6", "ov1000_t8", "ov2000_t8",
             "abl_tp5", "abl_spacing0", "abl_trail_fixed", "abl_gate_off", "abl_floor0", "abl_sl12.5",
             "peak2.2", "reentry0", "ov650_t8_sl12.5", "ov650_t8_oldlive", "sl0", "tp_classic"]
    order = first + [c for c in CONFIGS if c not in first]
    todo = [(a, c) for c in order for a in assets if (a, c) not in done]
    print(f"[grid] {len(assets)} assets x {len(CONFIGS)} configs, {len(todo)} tasks to run", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", 3))) as pool, open(OUT, "a") as out:
        for i, r in enumerate(pool.imap_unordered(task, todo), 1):
            out.write(json.dumps(r) + "\n"); out.flush()
            tag = "ERROR" if "error" in r else f"full {r['full']['ret']:+8.2f}% dd {r['full']['dd']:6.2f} ({r['secs']}s)"
            print(f"[grid] {i}/{len(todo)} {r['asset']:<9} {r['cfg']:<20} {tag}  [{time.time()-t0:.0f}s]", flush=True)
    print("[grid] done", flush=True)
