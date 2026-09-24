#!/usr/bin/env python3
"""Trading 212 strategy grid (runs on DEV): live profiles vs variants, continuous runs.

The live engine (212trading/strategy.py) through 212trading/replay.py, on Yahoo bars:
1h over the last ~730 days and 1d over ~10 years, one continuous run per dataset
(state carries across cycles, as live) plus fresh-state 90-day windows on 1h bars.

Each live profile (config.nvda/spcx/rgnt.env) runs on its own asset and on a basket of
US stocks/ETFs, to separate what is specific to one chart from what generalises.

Harness-only variant FLAGS["peak"]: after a take-profit, ratchet last_sell_price up to the
highest close since the sale, so re-entry waits for a pullback from the post-sale peak.
"""
import contextlib, csv, io, json, os, sys, time, traceback
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid")
DATA = os.path.join(EXP, "data")
OUT = os.path.join(EXP, "results.jsonl")
sys.path.insert(0, os.path.join(REPO, "212trading"))
sys.path.insert(0, REPO)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
import replay as rp                       # noqa: E402  (212trading/replay.py)
import strategy as strat                  # noqa: E402
from ipo_common import parse_dotenv       # noqa: E402

FLAGS = {"peak": False}
_orig_step = strat.Strategy.step


def _step(self, price):
    s = self.s
    if (FLAGS["peak"] and s["qty"] <= 1e-9 and s.get("last_sell_price")
            and not s.get("sl_rebuy")):
        s["last_sell_price"] = max(s["last_sell_price"], price)
    return _orig_step(self, price)


strat.Strategy.step = _step

PROFILES = {p: parse_dotenv(os.path.join(REPO, "212trading", f"config.{p}.env"))
            for p in ("nvda", "spcx", "rgnt")}
HOME = {"nvda": "NVDA", "spcx": "SPCX", "rgnt": "RGNT"}

VARIANTS = {
    "live": ({}, 0),
    "reentry0": ({"STRAT_REENTRY_DROP_PCT": "0"}, 0),
    "reentry5": ({"STRAT_REENTRY_DROP_PCT": "5"}, 0),
    "reentry_peak": ({}, 1),
    "tp3": ({"STRAT_TAKEPROFIT_PCT": "3", "STRAT_TP_LADDER": ""}, 0),
    "tp5": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": ""}, 0),
    "tp8": ({"STRAT_TAKEPROFIT_PCT": "8", "STRAT_TP_LADDER": ""}, 0),
    "tp12": ({"STRAT_TAKEPROFIT_PCT": "12", "STRAT_TP_LADDER": ""}, 0),
    "ladder": ({"STRAT_TP_LADDER": "11:33,20:33,30:34"}, 0),
    "dca3": ({"STRAT_DCA_DROP_PCT": "3"}, 0),
    "dca5": ({"STRAT_DCA_DROP_PCT": "5"}, 0),
    "sl0": ({"STRAT_STOP_LOSS_PCT": "0"}, 0),
    "sl15": ({"STRAT_STOP_LOSS_PCT": "15"}, 0),
    "trail8": ({"STRAT_TRAIL_PCT": "8", "STRAT_TRAIL_MIN_PROFIT_PCT": "9"}, 0),
    "trail_off": ({"STRAT_TRAIL_PCT": "0"}, 0),
    "tp5_peak": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": ""}, 1),
    "tp8_peak_sl0": ({"STRAT_TAKEPROFIT_PCT": "8", "STRAT_TP_LADDER": "", "STRAT_STOP_LOSS_PCT": "0"}, 1),
    "tp5_reentry0_sl0": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": "",
                          "STRAT_REENTRY_DROP_PCT": "0", "STRAT_STOP_LOSS_PCT": "0"}, 0),
}
# Factorial: take-profit x re-entry rule x stop-loss ("live" keeps the profile's value).
for _tp in ("1.5", "3", "5", "8", "12", "20"):
    for _re in ("live", "0", "peak"):
        for _sl in ("live", "0"):
            _over = {"STRAT_TAKEPROFIT_PCT": _tp, "STRAT_TP_LADDER": ""}
            if _re == "0":
                _over["STRAT_REENTRY_DROP_PCT"] = "0"
            if _sl == "0":
                _over["STRAT_STOP_LOSS_PCT"] = "0"
            VARIANTS[f"f_tp{_tp}_re{_re}_sl{_sl}"] = (_over, 1 if _re == "peak" else 0)
WARM, W90_1H, STEP_1H = 12, 90 * 7, 30 * 7   # ~7 regular-session 1h bars per day


def load(asset, itv):
    path = os.path.join(DATA, f"{asset}_{itv}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = [(int(r["timestamp"]), float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"])) for r in csv.DictReader(fh)]
    return rows if len(rows) > WARM + 20 else None


def params_for(profile, variant):
    cfg = dict(PROFILES[profile])
    over, peak = VARIANTS[variant]
    cfg.update(over)
    cfg["STRAT_DCA_TREND_GATE_PCT"] = "0"   # the gate needs 5m bars; neutral on 1h/1d
    return strat.StratParams.from_env(cfg), peak


def run(bars, params, peak, minutes):
    FLAGS["peak"] = bool(peak)
    with contextlib.redirect_stdout(io.StringIO()):
        m = rp.run_replay([b[1:] for b in bars[WARM:]], params, bar_minutes=minutes,
                          warmup_ohlc=[b[1:] for b in bars[:WARM]],
                          timestamps=[b[0] for b in bars[WARM:]])
    budget = float(params.max_budget)
    return {"ret": round(m["total"] / budget * 100, 3), "dd": round(m.get("max_drawdown_pct") or 0, 3),
            "cyc": m.get("cycles", 0), "expo": round(m.get("exposure_pct") or 0, 1),
            "fills": m.get("fills", 0),
            "bh": round((bars[-1][4] / bars[WARM][4] - 1) * 100, 2)}


def task(args):
    profile, variant, asset = args
    try:
        params, peak = params_for(profile, variant)
        out = {"profile": profile, "variant": variant, "asset": asset}
        t0 = time.time()
        for itv, minutes in (("1h", 60), ("1d", 1440)):
            bars = load(asset, itv)
            if bars:
                out[itv] = run(bars, params, peak, minutes)
        bars = load(asset, "1h")
        if bars:
            wins = []
            for s in range(WARM, len(bars) - W90_1H + 1, STEP_1H):
                seg = bars[s - WARM:s + W90_1H]
                r = run(seg, params, peak, 60)
                wins.append([r["ret"], r["dd"], r["bh"]])
            out["w90"] = wins
        out["secs"] = round(time.time() - t0, 1)
        return out
    except Exception:
        return {"profile": profile, "variant": variant, "asset": asset, "error": traceback.format_exc()}


if __name__ == "__main__":
    assets = sorted({f.split("_")[0] for f in os.listdir(DATA) if f.endswith(".csv")})
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
    print(f"[t212grid] {len(todo)} tasks", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", 2))) as pool, open(OUT, "w") as fh:
        for i, r in enumerate(pool.imap(task, todo), 1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            tag = "ERROR " + r["error"].strip().splitlines()[-1] if "error" in r else \
                f"1h {r.get('1h', {}).get('ret')} 1d {r.get('1d', {}).get('ret')} ({r.get('secs')}s)"
            print(f"[t212grid] {i}/{len(todo)} {r['profile']} {r['variant']:<16} {r['asset']:<6} {tag} [{time.time()-t0:.0f}s]", flush=True)
