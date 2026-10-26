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
from botcore import parse_dotenv  # noqa: E402
from strategies import spot_dca as strat  # noqa: E402
from kraken import replay as rp  # noqa: E402

_KRAKEN_CFG: dict[str, str] = {}
for _cfg_name in ("config.env", ".env"):
    _cfg_path = os.path.join(ROOT, "kraken", _cfg_name)
    if os.path.isfile(_cfg_path):
        _KRAKEN_CFG.update(parse_dotenv(_cfg_path))

FLAGS = {"peak": False}
_orig_step = strat.Strategy.step


def _step(self, price, *args, **kwargs):
    s = self.s
    if s["qty"] <= 1e-12:
        if FLAGS["peak"] and s.get("last_exit_kind") == "TP" and s.get("last_sell_price"):
            s["last_sell_price"] = max(s["last_sell_price"], price)
    return _orig_step(self, price, *args, **kwargs)


R = dataclasses.replace
LIVE = strat.StratParams.from_env(_KRAKEN_CFG)
OLD = R(LIVE, takeprofit_pct=5.0, dca_spacing_growth_pct=0.0, tp_trail_adaptive=False,
        tp_regime_gate=False, tp_trail_profit_floor_pct=0.0, stop_loss_pct=12.5,
        reentry_hybrid_enabled=False, reentry_bear_bounce_pct=0.0,
        tp_mdhtp_enabled=False, sl_mhdsl_enabled=False,
        trend_overlay=False, dca_trend_brake=False, surge_guard=False,
        fast_profit_guard=False, tp_dynamic_flat=False)


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
    "reentry0": (R(LIVE, reentry_drop_pct=0.0, reentry_hybrid_enabled=False), 0),
    "reentry1": (R(LIVE, reentry_drop_pct=1.0, reentry_hybrid_enabled=False), 0),
    "reentry4": (R(LIVE, reentry_drop_pct=4.0, reentry_hybrid_enabled=False), 0),
    "reentry_adaptive": (R(LIVE, reentry_adaptive=True, reentry_hybrid_enabled=False), 0),
    "peak2.2": (R(LIVE, reentry_hybrid_enabled=False), 1),
    "peak4": (R(LIVE, reentry_drop_pct=4.0, reentry_hybrid_enabled=False), 1),
    "peak2.2_oldlive": (OLD, 1),
    "ov650_t8_peak4": (ov(R(LIVE, reentry_drop_pct=4.0, reentry_hybrid_enabled=False), 650, 8), 1),
    # E. stop loss
    "sl0": (R(LIVE, stop_loss_pct=0.0), 0), "sl8": (R(LIVE, stop_loss_pct=8.0), 0),
    "sl25": (R(LIVE, stop_loss_pct=25.0), 0),
    # F. take profit
    "tp3": (R(LIVE, takeprofit_pct=3.0), 0), "tp6": (R(LIVE, takeprofit_pct=6.0), 0),
    "tp8": (R(LIVE, takeprofit_pct=8.0), 0),
    "tp_classic": (R(LIVE, tp_trend_hold=False, tp_mdhtp_enabled=False), 0),
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


_LOAD_CACHE: dict = {}
_ASSET_CACHE: dict = {}
_TASK_CACHE: dict = {}


def load(path):
    cached = _LOAD_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path) as fh:
        rows = [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"])) for r in csv.DictReader(fh)]
    _LOAD_CACHE[path] = rows
    return rows


def load_asset(asset):
    cached = _ASSET_CACHE.get(asset)
    if cached is not None:
        return cached
    bars = load(os.path.join(DATA, f"{asset}_240m.csv"))
    ohlc = [b[1:] for b in bars]
    if len(ohlc) > WARM:
        rp.prewarm_trend_series(ohlc, warmup=WARM, params=LIVE, bar_minutes=240.0)
    bar_years = [datetime.fromtimestamp(b[0], timezone.utc).year for b in bars]
    years = sorted(set(bar_years))
    year_slices = []
    for y in years:
        idx = [i for i, yr in enumerate(bar_years) if yr == y]
        if len(idx) >= 360 and idx[0] >= WARM:
            year_slices.append((str(y), idx[0], idx[-1] + 1))
    res = (bars, ohlc, year_slices)
    _ASSET_CACHE[asset] = res
    return res


def replay(bars, warm, params, peak, fee=0.26, initial_cash=None):
    FLAGS["peak"] = bool(peak)
    strat.Strategy.step = _step if peak else _orig_step
    try:
        ohlc = [b[1:] for b in bars] if (bars and len(bars[0]) == 5) else bars
        w_ohlc = [b[1:] for b in warm] if (warm and len(warm[0]) == 5) else warm
        if not ohlc:
            if initial_cash is not None:
                return {"ret": 0.0, "dd": 0.0, "cyc": 0, "refused": 0, "min_cash": float(initial_cash)}
            return {"ret": 0.0, "dd": 0.0, "cyc": 0, "expo": 0.0, "bh": 0.0}
        m = rp.run_replay(ohlc, params, fee_pct=fee, bar_minutes=240,
                          warmup_ohlc=w_ohlc, initial_cash=initial_cash)
        tot = float(m.get("total") or 0.0)
        if initial_cash is not None:
            ic = float(initial_cash)
            funding = m.get("funding") or {}
            return {"ret": round(tot / ic * 100, 2) if ic > 0 else 0.0,
                    "dd": round(m.get("max_drawdown_pct") or 0, 2),
                    "cyc": int(m.get("cycles") or 0),
                    "refused": int(funding.get("refused_buys") or 0),
                    "min_cash": float(funding.get("minimum_cash") if funding.get("minimum_cash") is not None else ic)}
        budget = float(params.effective_max_budget())
        p0 = float(ohlc[0][3])
        p1 = float(ohlc[-1][3])
        return {"ret": round(tot / budget * 100, 3) if budget > 0 else 0.0,
                "dd": round(m.get("max_drawdown_pct") or 0, 3),
                "cyc": int(m.get("cycles") or 0),
                "expo": round(m.get("exposure_pct") or 0, 1),
                "bh": round((p1 / p0 - 1) * 100, 3) if p0 > 0 else 0.0}
    finally:
        FLAGS["peak"] = False
        strat.Strategy.step = _orig_step


def task(args):
    asset, cfg = args
    try:
        bars, ohlc, year_slices = load_asset(asset)
        params, peak = CONFIGS[cfg]
        cache_key = (asset, len(bars), bars[0][0] if bars else 0, bars[-1][0] if bars else 0,
                     repr(params), bool(peak))
        cached = _TASK_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cfg": cfg, "secs": 0.0}
        t0 = time.time()
        res = {"asset": asset, "cfg": cfg, "bars": len(bars),
               "full": replay(ohlc[WARM:], ohlc[:WARM], params, peak), "year": {}, "w30": [], "w90": []}
        for y_str, s_idx, e_idx in year_slices:
            res["year"][y_str] = replay(ohlc[s_idx:e_idx], ohlc[s_idx - WARM:s_idx], params, peak)
        for key, size in (("w30", W30), ("w90", W90)):
            for s in range(WARM, len(ohlc) - size + 1, WSTEP):
                r = replay(ohlc[s:s + size], ohlc[s - WARM:s], params, peak)
                res[key].append([r["ret"], r["dd"], r["bh"], r["cyc"]])
        res["secs"] = round(time.time() - t0, 1)
        _TASK_CACHE[cache_key] = res
        return res
    except Exception:
        return {"asset": asset, "cfg": cfg, "error": traceback.format_exc()}


if __name__ == "__main__":
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    for a in assets:
        load_asset(a)
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["asset"], r["cfg"]))
    first = ["live", "old_live_pre0923", "ov650_t8", "ov350_t6", "ov1000_t8", "ov2000_t8",
             "abl_tp5", "abl_spacing0", "abl_trail_fixed", "abl_gate_off", "abl_floor0", "abl_sl12.5",
             "peak2.2", "reentry0", "ov650_t8_sl12.5", "ov650_t8_oldlive", "sl0", "tp_classic"]
    order = first + [c for c in CONFIGS if c not in first]
    todo = [(a, c) for a in assets for c in order if (a, c) not in done]
    canon_groups: dict = {}
    canon_todo = []
    for a, c in todo:
        p, pk = CONFIGS[c]
        k = (a, repr(p), bool(pk))
        if k not in canon_groups:
            canon_groups[k] = []
            canon_todo.append((a, c))
        canon_groups[k].append(c)
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 4))
    print(f"[grid] {len(assets)} assets x {len(CONFIGS)} configs, {len(todo)} tasks ({len(canon_todo)} unique) on {workers} workers", flush=True)
    t0 = time.time()
    done_count = 0
    with Pool(workers) as pool, open(OUT, "a") as out:
        for r in pool.imap_unordered(task, canon_todo):
            p, pk = CONFIGS[r["cfg"]]
            aliases = canon_groups.get((r["asset"], repr(p), bool(pk)), [r["cfg"]])
            for idx, alias_cfg in enumerate(aliases):
                done_count += 1
                row = r if idx == 0 and alias_cfg == r["cfg"] else {**r, "cfg": alias_cfg, "secs": 0.0}
                out.write(json.dumps(row) + "\n")
                tag = "ERROR" if "error" in row else f"full {row['full']['ret']:+8.2f}% dd {row['full']['dd']:6.2f} ({row['secs']}s)"
                print(f"[grid] {done_count}/{len(todo)} {row['asset']:<9} {row['cfg']:<20} {tag}  [{time.time()-t0:.0f}s]", flush=True)
            out.flush()
    print("[grid] done", flush=True)
