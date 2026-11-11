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
import os, sys, time, traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import lru_dict_put, safe_float, safe_pos_float
from state_io import append_jsonl_record, atomic_write_jsonl, read_jsonl_dicts
from offline.research.kraken_continuous_grid import grid
from offline.research.kraken_continuous_grid.report import _risk_adjusted_score
CONFIGS, LIVE, OLD, R, ov, load = grid.CONFIGS, grid.LIVE, grid.OLD, grid.R, grid.ov, grid.load
WARM, W30, W90, WSTEP, DATA, EXP = grid.WARM, grid.W30, grid.W90, grid.WSTEP, grid.DATA, grid.EXP

OUT = os.path.join(EXP, "results2.jsonl")
FEE = {"std": 0.26, "stress": 0.40}

FINE = {
    **{f"ov{t}_t{tr}": (ov(LIVE, t, tr), 0) for t in (500, 800, 1300) for tr in (7, 9, 12)},
    "ov650_t8_gateoff": (ov(R(LIVE, tp_regime_gate=False), 650, 8), 0),
    "ov650_t8_tp5": (ov(R(LIVE, takeprofit_pct=5.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 650, 8), 0),
    "ov650_t8_sl25": (ov(R(LIVE, stop_loss_pct=25.0), 650, 8), 0),
    "ov650_t8_sl0": (ov(R(LIVE, stop_loss_pct=0.0), 650, 8), 0),
    "ov650_t8_reentry4": (ov(R(LIVE, reentry_drop_pct=4.0, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 650, 8), 0),
    "ov650_t8_dca1.5": (ov(R(LIVE, dca_drop_pct=1.5), 650, 8), 0),
    "ov1000_t10_gateoff": (ov(R(LIVE, tp_regime_gate=False), 1000, 10), 0),
    "ov1000_t8_gateoff_tp5": (ov(R(LIVE, tp_regime_gate=False, takeprofit_pct=5.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 1000, 8), 0),
    "gateoff_tp5": (R(LIVE, tp_regime_gate=False, takeprofit_pct=5.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 0),
    "gateoff_trail5": (R(LIVE, tp_regime_gate=False, tp_mdhtp_enabled=False, tp_trail_adaptive=False, tp_trail_pct=5.0), 0),
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


_VIEWS_CACHE: dict = {}


replay = grid.replay


def views(asset, cfg, fee_key):
    bars, ohlc, _ = grid.load_asset(asset)
    params, peak = ALL[cfg]
    start_ts = bars[0][0] if bars else 0
    end_ts = bars[-1][0] if bars else 0
    cache_key = (os.path.abspath(grid.DATA), asset, len(bars), start_ts, end_ts,
                 repr(params), bool(peak), fee_key)
    cached = _VIEWS_CACHE.get(cache_key)
    if cached is not None:
        return {**cached, "cfg": cfg}
    fee = FEE[fee_key]
    res = {"kind": "views", "fee": fee_key, "asset": asset, "cfg": cfg, "bars": len(bars),
           "start_ts": start_ts, "end_ts": end_ts,
           "params_repr": repr(params), "peak": bool(peak),
           "full": replay(bars[WARM:], bars[:WARM], params, peak, fee), "w30": [], "w90": []}
    for key, size in (("w30", W30), ("w90", W90)):
        for s in range(WARM, len(bars) - size + 1, WSTEP):
            r = replay(bars[s:s + size], bars[s - WARM:s], params, peak, fee)
            res[key].append([r["ret"], r["dd"], r["bh"], r["cyc"]])
    lru_dict_put(_VIEWS_CACHE, cache_key, res, max_entries=512)
    return res


def select(asset):
    bars, ohlc, _ = grid.load_asset(asset)
    steps = []
    fee_r = 0.26 / 100.0
    for t in range(WARM + TRAIN, len(bars) - TEST + 1, TEST):
        train_slice = bars[t - TRAIN:t]
        train_warm = bars[t - TRAIN - WARM:t - TRAIN]
        train_cache = {}
        train = {}
        for c in SELECT_SET:
            p, pk = ALL[c]
            k = (repr(p), bool(pk))
            if k not in train_cache:
                train_cache[k] = replay(train_slice, train_warm, p, pk, 0.26)
            train[c] = train_cache[k]
        score = {c: _risk_adjusted_score(r["ret"], r["dd"], dd_floor=5.0) for c, r in train.items()}
        best = max(score, key=score.get)
        oos_slice = bars[t:t + TEST]
        oos_warm = bars[t - WARM:t]
        oos_cache = {}
        oos = {}
        for c in SELECT_SET:
            p, pk = ALL[c]
            k = (repr(p), bool(pk))
            if k not in oos_cache:
                oos_cache[k] = replay(oos_slice, oos_warm, p, pk, 0.26)["ret"]
            oos[c] = oos_cache[k]
        p0 = safe_pos_float(ohlc[t][3], 0.0)
        p1 = safe_float(ohlc[t + TEST - 1][3], 0.0)
        bh_pct = (((p1 * (1.0 - fee_r)) / (p0 * (1.0 + fee_r))) - 1.0) * 100.0 if p0 > 0 else 0.0
        steps.append({"t": bars[t][0], "pick": best, "pick_ret": oos[best], "oos": oos,
                      "bh": round(bh_pct, 2)})
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
    for a in assets:
        grid.load_asset(a)
    for r in read_jsonl_dicts(grid.OUT):
        if "error" not in r and r.get("cfg") in ALL and r.get("asset") in assets:
            bars, _, _ = grid.load_asset(r["asset"])
            start_ts = bars[0][0] if bars else 0
            end_ts = bars[-1][0] if bars else 0
            p, pk = ALL[r["cfg"]]
            if (
                len(bars) == r.get("bars")
                and isinstance(r.get("full"), dict)
                and isinstance(r.get("w30"), list)
                and isinstance(r.get("w90"), list)
                and "start_ts" in r and r.get("start_ts") == start_ts
                and "end_ts" in r and r.get("end_ts") == end_ts
                and "params_repr" in r and r.get("params_repr") == repr(p)
                and "peak" in r and bool(r.get("peak")) == bool(pk)
            ):
                ck = (
                    os.path.abspath(grid.DATA), r["asset"], len(bars), start_ts, end_ts,
                    repr(p), bool(pk), "std",
                )
                lru_dict_put(
                    _VIEWS_CACHE,
                    ck,
                    {
                        "kind": "views",
                        "fee": "std",
                        "asset": r["asset"],
                        "cfg": r["cfg"],
                        "bars": r["bars"],
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "params_repr": repr(p),
                        "peak": bool(pk),
                        "full": r["full"],
                        "w30": r["w30"],
                        "w90": r["w90"],
                    },
                    max_entries=512,
                )
    assets_by_len = sorted(assets, key=lambda a: -len(grid.load_asset(a)[0]))
    todo = []
    for a in assets_by_len:
        todo.append(("select", a))
    for a in assets_by_len:
        for c in FINE:
            todo.append(("views", a, c, "std"))
        for c in FEE_SET:
            todo.append(("views", a, c, "stress"))
    canon_groups: dict = {}
    canon_todo = []
    for item in todo:
        if item[0] == "select":
            canon_todo.append(item)
        else:
            _, a, c, fk = item
            p, pk = ALL[c]
            k = (a, repr(p), bool(pk), fk)
            if k not in canon_groups:
                canon_groups[k] = []
                canon_todo.append(item)
            canon_groups[k].append(c)
    workers = grid.resolve_grid_workers()
    print(f"[grid2] {len(todo)} tasks ({len(canon_todo)} unique) on {workers} workers", flush=True)
    t0 = time.time()
    done_count = 0
    atomic_write_jsonl(OUT, ())
    with Pool(workers) as pool:
        for r in pool.imap_unordered(task, canon_todo):
            tag = r.get("kind")
            if tag == "views":
                p, pk = ALL[r["cfg"]]
                aliases = canon_groups.get((r["asset"], repr(p), bool(pk), r["fee"]), [r["cfg"]])
                for idx, alias_cfg in enumerate(aliases):
                    done_count += 1
                    row = r if idx == 0 and alias_cfg == r["cfg"] else {**r, "cfg": alias_cfg, "secs": 0.0}
                    append_jsonl_record(OUT, row)
                    vtag = f"{row['asset']:<5} {row['cfg']:<24} {row['fee']:<6} full {row['full']['ret']:+8.2f}% dd {row['full']['dd']:6.2f}"
                    print(f"[grid2] {done_count}/{len(todo)} {vtag} [{time.time()-t0:.0f}s]", flush=True)
            elif tag == "error" and isinstance(r.get("args"), list) and len(r["args"]) == 4 and r["args"][0] == "views":
                _, err_a, err_c, err_fk = r["args"]
                p, pk = ALL[err_c]
                aliases = canon_groups.get((err_a, repr(p), bool(pk), err_fk), [err_c])
                for alias_cfg in aliases:
                    done_count += 1
                    err_row = {**r, "asset": err_a, "cfg": alias_cfg, "fee": err_fk, "args": ["views", err_a, alias_cfg, err_fk]}
                    append_jsonl_record(OUT, err_row)
                    print(f"[grid2] {done_count}/{len(todo)} ERROR {err_a} {alias_cfg} {err_fk} [{time.time()-t0:.0f}s]", flush=True)
            else:
                done_count += 1
                append_jsonl_record(OUT, r)
                if tag == "select":
                    tag = f"select {r['asset']} ({len(r['steps'])} steps)"
                print(f"[grid2] {done_count}/{len(todo)} {tag} [{time.time()-t0:.0f}s]", flush=True)
    print("[grid2] done", flush=True)
