#!/usr/bin/env python3
"""Trading 212 strategy grid (runs on DEV): live profiles vs variants, continuous runs.

The live engine (212trading/strategy.py) through 212trading/replay.py, on Yahoo bars:
1h since Oct 2023 (Yahoo range=730d, ~3 years) and 1d over ~10 years, one continuous run per dataset
(state carries across cycles, as live) plus fresh-state 90-day windows on 1h bars.

Each live profile (config.nvda/spcx/rgnt.env) runs on its own asset and on a basket of
US stocks/ETFs, to separate what is specific to one chart from what generalises.

Harness-only variant FLAGS["peak"]: after a take-profit, ratchet last_sell_price up to the
highest close since the sale, so re-entry waits for a pullback from the post-sale peak.
"""
import contextlib, csv, importlib.util, io, json, os, sys, time, traceback
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid")
DATA = os.path.join(EXP, "data")
OUT = os.path.join(EXP, "results.jsonl")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
_orig_sys_path = list(sys.path)
try:
    _rp_spec = importlib.util.spec_from_file_location("t212_replay_grid", os.path.join(REPO, "212trading", "replay.py"))
    rp = importlib.util.module_from_spec(_rp_spec)
    sys.modules[_rp_spec.name] = rp
    _rp_spec.loader.exec_module(rp)
    strat = rp._strat
    from botcore import parse_dotenv       # noqa: E402
finally:
    sys.path[:] = _orig_sys_path

FLAGS = {"peak": False}
_orig_step = strat.Strategy.step


def _step(self, price, *args, **kwargs):
    s = self.s
    if (FLAGS["peak"] and s["qty"] <= 1e-9 and s.get("last_sell_price")
            and not s.get("sl_rebuy")):
        s["last_sell_price"] = max(s["last_sell_price"], price)
    return _orig_step(self, price, *args, **kwargs)


PROFILES = {p: parse_dotenv(os.path.join(REPO, "212trading", f"config.{p}.env"))
            for p in ("nvda", "spcx", "rgnt")}
HOME = {"nvda": "NVDA", "spcx": "SPCX", "rgnt": "RGNT"}

VARIANTS = {
    "live": ({}, 0),
    "reentry0": ({"STRAT_REENTRY_DROP_PCT": "0", "STRAT_REENTRY_HYBRID_ENABLED": "false"}, 0),
    "reentry5": ({"STRAT_REENTRY_DROP_PCT": "5", "STRAT_REENTRY_HYBRID_ENABLED": "false"}, 0),
    "reentry_peak": ({"STRAT_REENTRY_HYBRID_ENABLED": "false"}, 1),
    "tp3": ({"STRAT_TAKEPROFIT_PCT": "3", "STRAT_TP_LADDER": ""}, 0),
    "tp5": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": ""}, 0),
    "tp8": ({"STRAT_TAKEPROFIT_PCT": "8", "STRAT_TP_LADDER": ""}, 0),
    "tp12": ({"STRAT_TAKEPROFIT_PCT": "12", "STRAT_TP_LADDER": ""}, 0),
    "ladder": ({"STRAT_TP_LADDER": "11:33,20:33,30:34", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "dca3": ({"STRAT_DCA_DROP_PCT": "3"}, 0),
    "dca5": ({"STRAT_DCA_DROP_PCT": "5"}, 0),
    "sl0": ({"STRAT_STOP_LOSS_PCT": "0"}, 0),
    "sl15": ({"STRAT_STOP_LOSS_PCT": "15"}, 0),
    "trail8": ({"STRAT_TRAIL_PCT": "8", "STRAT_TRAIL_MIN_PROFIT_PCT": "9"}, 0),
    "trail_off": ({"STRAT_TRAIL_PCT": "0"}, 0),
    "tp5_peak": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": "", "STRAT_REENTRY_HYBRID_ENABLED": "false"}, 1),
    "tp8_peak_sl0": ({"STRAT_TAKEPROFIT_PCT": "8", "STRAT_TP_LADDER": "", "STRAT_STOP_LOSS_PCT": "0", "STRAT_REENTRY_HYBRID_ENABLED": "false"}, 1),
    "tp5_reentry0_sl0": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": "",
                          "STRAT_REENTRY_DROP_PCT": "0", "STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_STOP_LOSS_PCT": "0"}, 0),
}
# Factorial: take-profit x re-entry rule x stop-loss ("live" keeps the profile's value).
for _tp in ("1.5", "3", "5", "8", "12", "20"):
    for _re in ("live", "0", "peak"):
        for _sl in ("live", "0"):
            _over = {"STRAT_TAKEPROFIT_PCT": _tp, "STRAT_TP_LADDER": ""}
            if _re == "0":
                _over["STRAT_REENTRY_DROP_PCT"] = "0"
                _over["STRAT_REENTRY_HYBRID_ENABLED"] = "false"
            elif _re == "peak":
                _over["STRAT_REENTRY_HYBRID_ENABLED"] = "false"
            if _sl == "0":
                _over["STRAT_STOP_LOSS_PCT"] = "0"
            VARIANTS[f"f_tp{_tp}_re{_re}_sl{_sl}"] = (_over, 1 if _re == "peak" else 0)
WARM, W90_1H, STEP_1H = 12, 90 * 7, 30 * 7   # ~7 regular-session 1h bars per day


_LOAD_CACHE: dict = {}
_SPLIT_CACHE: dict = {}
_TASK_CACHE: dict = {}


def load(asset, itv):
    key = (asset, itv)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    path = os.path.join(DATA, f"{asset}_{itv}.csv")
    if not os.path.exists(path):
        _LOAD_CACHE[key] = None
        _SPLIT_CACHE[key] = (None, None)
        return None
    import math
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ts_val = int(float(r["timestamp"]))
                o = float(r["open"])
                h = float(r["high"])
                l = float(r["low"])
                c = float(r["close"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if ts_val > 0 and all(math.isfinite(x) and x > 0 for x in (o, h, l, c)):
                rows.append((ts_val, o, h, l, c))
    res = rows if len(rows) > WARM + 20 else None
    _LOAD_CACHE[key] = res
    if res is not None:
        _SPLIT_CACHE[key] = ([b[1:] for b in res], [b[0] for b in res])
    else:
        _SPLIT_CACHE[key] = (None, None)
    return res


def params_for(profile, variant):
    cfg = dict(PROFILES[profile])
    over, peak = VARIANTS[variant]
    cfg.update(over)
    if peak and float(cfg.get("STRAT_REENTRY_DROP_PCT") or 0) <= 0:
        pb = float(cfg.get("STRAT_REENTRY_PULLBACK_PCT") or 0)
        cfg["STRAT_REENTRY_DROP_PCT"] = str(pb if pb > 0 else 2.0)
    cfg["STRAT_DCA_TREND_GATE_PCT"] = "0"   # the gate needs 5m bars; neutral on 1h/1d
    return strat.StratParams.from_env(cfg), peak


def run(bars, params, peak, minutes, ohlc=None, ts=None):
    FLAGS["peak"] = bool(peak)
    strat.Strategy.step = _step if peak else _orig_step
    try:
        if ohlc is None:
            ohlc = [b[1:] for b in bars]
        if ts is None:
            ts = [b[0] for b in bars]
        if len(ohlc) <= WARM:
            return {"ret": 0.0, "dd": 0.0, "cyc": 0, "expo": 0.0, "fills": 0, "bh": 0.0}
        m = rp.run_replay(ohlc[WARM:], params, bar_minutes=minutes,
                          warmup_ohlc=ohlc[:WARM],
                          timestamps=ts[WARM:])
        budget = float(params.max_budget)
        tot = float(m.get("total") or 0.0)
        p0 = float(ohlc[WARM][3])
        p1 = float(ohlc[-1][3])
        return {"ret": round(tot / budget * 100, 3) if budget > 0 else 0.0,
                "dd": round(m.get("max_drawdown_pct") or 0, 3),
                "cyc": int(m.get("cycles") or 0),
                "expo": round(m.get("exposure_pct") or 0, 1),
                "fills": int(m.get("fills") or 0),
                "bh": round((p1 / p0 - 1) * 100, 2) if p0 > 0 else 0.0}
    finally:
        FLAGS["peak"] = False
        strat.Strategy.step = _orig_step


def task(args):
    profile, variant, asset = args
    try:
        params, peak = params_for(profile, variant)
        bars_1h = load(asset, "1h")
        bars_1d = load(asset, "1d")
        cache_key = (
            asset,
            len(bars_1h) if bars_1h else 0,
            len(bars_1d) if bars_1d else 0,
            repr(params),
            bool(peak),
        )
        cached = _TASK_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "profile": profile, "variant": variant, "secs": 0.0}
        out = {"profile": profile, "variant": variant, "asset": asset}
        t0 = time.time()
        for itv, minutes, bars in (("1h", 60, bars_1h), ("1d", 1440, bars_1d)):
            if bars:
                ohlc, ts = _SPLIT_CACHE.get((asset, itv), (None, None))
                out[itv] = run(bars, params, peak, minutes, ohlc=ohlc, ts=ts)
        if bars_1h:
            ohlc, ts = _SPLIT_CACHE.get((asset, "1h"), (None, None))
            if ohlc is None:
                ohlc = [b[1:] for b in bars_1h]
                ts = [b[0] for b in bars_1h]
            wins = []
            for s in range(WARM, len(bars_1h) - W90_1H + 1, STEP_1H):
                r = run(
                    None, params, peak, 60,
                    ohlc=ohlc[s - WARM:s + W90_1H],
                    ts=ts[s - WARM:s + W90_1H],
                )
                wins.append([r["ret"], r["dd"], r["bh"]])
            out["w90"] = wins
        out["secs"] = round(time.time() - t0, 1)
        _TASK_CACHE[cache_key] = out
        return out
    except Exception:
        return {"profile": profile, "variant": variant, "asset": asset, "error": traceback.format_exc()}


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(EXP, exist_ok=True)
    assets = sorted({f.split("_")[0] for f in os.listdir(DATA) if f.endswith(".csv")})
    if not assets:
        print(f"[t212grid] No CSV data files found in {DATA}. Run fetch.py first.", flush=True)
        sys.exit(0)
    for a in assets:
        load(a, "1h")
        load(a, "1d")
    todo = []
    for profile in PROFILES:
        home = HOME[profile]
        for variant in VARIANTS:
            todo.append((profile, variant, home))
    for profile in PROFILES:            # generalisation: every profile on every asset
        for variant in VARIANTS:
            for asset in assets:
                if asset != HOME[profile]:
                    todo.append((profile, variant, asset))
    canon_groups: dict = {}
    canon_todo = []
    for profile, variant, asset in todo:
        p, pk = params_for(profile, variant)
        k = (asset, repr(p), bool(pk))
        if k not in canon_groups:
            canon_groups[k] = []
            canon_todo.append((profile, variant, asset))
        canon_groups[k].append((profile, variant))
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 4))
    print(f"[t212grid] {len(todo)} tasks ({len(canon_todo)} unique) on {workers} workers", flush=True)
    t0 = time.time()
    done_count = 0
    with Pool(workers) as pool, open(OUT, "w") as fh:
        for r in pool.imap(task, canon_todo):
            p, pk = params_for(r["profile"], r["variant"])
            aliases = canon_groups.get((r["asset"], repr(p), bool(pk)), [(r["profile"], r["variant"])])
            for idx, (al_prof, al_var) in enumerate(aliases):
                done_count += 1
                row = r if idx == 0 and (al_prof, al_var) == (r["profile"], r["variant"]) else {
                    **r, "profile": al_prof, "variant": al_var, "secs": 0.0
                }
                fh.write(json.dumps(row) + "\n")
                err_lines = str(row.get("error") or "").strip().splitlines()
                tag = ("ERROR " + (err_lines[-1] if err_lines else "unknown")) if "error" in row else \
                    f"1h {row.get('1h', {}).get('ret')} 1d {row.get('1d', {}).get('ret')} ({row.get('secs')}s)"
                print(f"[t212grid] {done_count}/{len(todo)} {row['profile']} {row['variant']:<16} {row['asset']:<6} {tag} [{time.time()-t0:.0f}s]", flush=True)
            fh.flush()
