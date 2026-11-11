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
from bisect import bisect_left
import os, sys, time, traceback
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from botcore import lru_dict_put, optional_env, parse_dotenv, safe_float, safe_pos_float  # noqa: E402
EXP = optional_env("GRID_OUT") or os.path.join(REPO, "offline", "results", "t212_continuous_grid")
DATA = os.path.join(EXP, "data")
OUT = os.path.join(EXP, "results.jsonl")
from offline.backtests.datasets import load_ohlc_tuples_csv  # noqa: E402
from offline.backtests.evaluation import load_venue_replay_module  # noqa: E402
from offline.research.kraken_continuous_grid.grid import resolve_grid_workers  # noqa: E402
from state_io import atomic_write_jsonl  # noqa: E402
from strategies.spot_replay import (  # noqa: E402
    post_exit_peak_step_scope,
    prepare_spot_interval_params,
    ratchet_post_exit_peak_sell_price,
    summarize_grid_replay_metrics,
)

os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
rp = load_venue_replay_module("t212", "t212_replay_grid")
strat = rp._strat

FLAGS = {"peak": False}
_orig_step = strat.Strategy.step


def _step(self, price, *args, **kwargs):
    if FLAGS["peak"]:
        ratchet_post_exit_peak_sell_price(self.s, price, qty_eps=1e-9, require_tp_kind=True)
    return _orig_step(self, price, *args, **kwargs)


PROFILES = {p: parse_dotenv(os.path.join(REPO, "212trading", f"config.{p}.env"))
            for p in ("nvda", "spcx", "rgnt")}
HOME = {"nvda": "NVDA", "spcx": "SPCX", "rgnt": "RGNT"}

VARIANTS = {
    "live": ({}, 0),
    "reentry0": ({"STRAT_REENTRY_DROP_PCT": "0", "STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_REENTRY_TTL_HOURS": "0"}, 0),
    "reentry5": ({"STRAT_REENTRY_DROP_PCT": "5", "STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_REENTRY_TTL_HOURS": "0"}, 0),
    "reentry_peak": ({"STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_REENTRY_TTL_HOURS": "0"}, 1),
    "tp3": ({"STRAT_TAKEPROFIT_PCT": "3", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "tp5": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "tp8": ({"STRAT_TAKEPROFIT_PCT": "8", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "tp12": ({"STRAT_TAKEPROFIT_PCT": "12", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "ladder": ({"STRAT_TP_LADDER": "11:33,20:33,30:34", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "dca3": ({"STRAT_DCA_DROP_PCT": "3"}, 0),
    "dca5": ({"STRAT_DCA_DROP_PCT": "5"}, 0),
    "sl0": ({"STRAT_STOP_LOSS_PCT": "0"}, 0),
    "sl15": ({"STRAT_STOP_LOSS_PCT": "15"}, 0),
    "trail8": ({"STRAT_TRAIL_PCT": "8", "STRAT_TRAIL_MIN_PROFIT_PCT": "9", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "trail_off": ({"STRAT_TRAIL_PCT": "0", "STRAT_TP_MDHTP_ENABLED": "false"}, 0),
    "tp5_peak": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false", "STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_REENTRY_TTL_HOURS": "0"}, 1),
    "tp8_peak_sl0": ({"STRAT_TAKEPROFIT_PCT": "8", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false", "STRAT_STOP_LOSS_PCT": "0", "STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_REENTRY_TTL_HOURS": "0"}, 1),
    "tp5_reentry0_sl0": ({"STRAT_TAKEPROFIT_PCT": "5", "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false",
                          "STRAT_REENTRY_DROP_PCT": "0", "STRAT_REENTRY_HYBRID_ENABLED": "false", "STRAT_REENTRY_TTL_HOURS": "0", "STRAT_STOP_LOSS_PCT": "0"}, 0),
}
# Factorial: take-profit x re-entry rule x stop-loss ("live" keeps the profile's value).
for _tp in ("1.5", "3", "5", "8", "12", "20"):
    for _re in ("live", "0", "peak"):
        for _sl in ("live", "0"):
            _over = {"STRAT_TAKEPROFIT_PCT": _tp, "STRAT_TP_LADDER": "", "STRAT_TP_MDHTP_ENABLED": "false"}
            if _re == "0":
                _over["STRAT_REENTRY_DROP_PCT"] = "0"
                _over["STRAT_REENTRY_HYBRID_ENABLED"] = "false"
                _over["STRAT_REENTRY_TTL_HOURS"] = "0"
            elif _re == "peak":
                _over["STRAT_REENTRY_HYBRID_ENABLED"] = "false"
                _over["STRAT_REENTRY_TTL_HOURS"] = "0"
            if _sl == "0":
                _over["STRAT_STOP_LOSS_PCT"] = "0"
            VARIANTS[f"f_tp{_tp}_re{_re}_sl{_sl}"] = (_over, 1 if _re == "peak" else 0)
WARM, W90_1H, STEP_1H = 20, 90 * 7, 30 * 7   # ~7 regular-session 1h bars per day
W90_SEC, STEP30_SEC = 90 * 86400, 30 * 86400


_LOAD_CACHE: dict = {}
_SPLIT_CACHE: dict = {}
_TASK_CACHE: dict = {}


def _extract_asset_name(filename: str) -> str | None:
    if not isinstance(filename, str) or not filename.endswith(".csv"):
        return None
    for suffix in ("_1h.csv", "_1d.csv"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)].strip()
            return stem or None
    stem = filename[:-4].strip()
    return stem or None


def load(asset, itv):
    key = (DATA, asset, itv)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    path = os.path.join(DATA, f"{asset}_{itv}.csv")
    if not os.path.exists(path):
        _LOAD_CACHE[key] = None
        _SPLIT_CACHE[key] = (None, None)
        return None
    rows = load_ohlc_tuples_csv(path, strict=False)
    res = rows if len(rows) > WARM + 12 else None
    _LOAD_CACHE[key] = res
    if res is not None:
        ohlc_list = [b[1:] for b in res]
        _SPLIT_CACHE[key] = (ohlc_list, [b[0] for b in res])
    else:
        _SPLIT_CACHE[key] = (None, None)
    return res


def params_for(profile, variant):
    cfg = dict(PROFILES[profile])
    over, peak = VARIANTS[variant]
    cfg.update(over)
    if peak and safe_float(cfg.get("STRAT_REENTRY_DROP_PCT"), 0.0) <= 0:
        pb = safe_pos_float(cfg.get("STRAT_REENTRY_PULLBACK_PCT"), default=2.0)
        cfg["STRAT_REENTRY_DROP_PCT"] = str(pb)
    cfg["STRAT_DCA_TREND_GATE_PCT"] = "0"   # the gate needs 5m bars; neutral on 1h/1d
    return strat.StratParams.from_env(cfg), peak


def run(bars, params, peak, minutes, ohlc=None, ts=None):
    with post_exit_peak_step_scope(strat.Strategy, FLAGS, peak, _step, _orig_step):
        if ohlc is None:
            ohlc = [b[1:] for b in bars]
        if ts is None:
            ts = [b[0] for b in bars]
        if len(ohlc) <= WARM:
            return {"ret": 0.0, "dd": 0.0, "cyc": 0, "expo": 0.0, "fills": 0, "bh": 0.0}
        interval_params = prepare_spot_interval_params(params, minutes)
        budget = safe_pos_float(params.max_budget, 0.0)
        m = rp.run_replay(ohlc[WARM:], interval_params, bar_minutes=minutes,
                          warmup_ohlc=ohlc[:WARM],
                          timestamps=ts[WARM:],
                          warmup_timestamps=ts[:WARM],
                          initial_cash=(budget if budget > 0.0 else None))
        return summarize_grid_replay_metrics(
            m,
            budget=budget,
            p0=ohlc[WARM][3],
            p1=ohlc[-1][3],
            fee_pct=getattr(params, "fx_fee_pct", 0.0),
            bh_digits=2,
            include_fills=True,
        )


def task(args):
    profile, variant, asset = args
    try:
        params, peak = params_for(profile, variant)
        bars_1h = load(asset, "1h")
        bars_1d = load(asset, "1d")
        span_1h = (len(bars_1h), bars_1h[0][0], bars_1h[-1][0]) if bars_1h else (0, 0, 0)
        span_1d = (len(bars_1d), bars_1d[0][0], bars_1d[-1][0]) if bars_1d else (0, 0, 0)
        cache_key = (
            DATA,
            asset,
            span_1h,
            span_1d,
            repr(params),
            bool(peak),
        )
        cached = _TASK_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "profile": profile, "variant": variant, "secs": 0.0}
        out = {
            "profile": profile,
            "variant": variant,
            "asset": asset,
            "params_repr": repr(params),
            "peak": bool(peak),
        }
        t0 = time.time()
        for itv, minutes, bars in (("1h", 60, bars_1h), ("1d", 1440, bars_1d)):
            if bars:
                ohlc, ts = _SPLIT_CACHE.get((DATA, asset, itv), (None, None))
                out[itv] = run(bars, params, peak, minutes, ohlc=ohlc, ts=ts)
        if bars_1h:
            ohlc, ts = _SPLIT_CACHE.get((DATA, asset, "1h"), (None, None))
            if ohlc is None:
                ohlc = [b[1:] for b in bars_1h]
                ts = [b[0] for b in bars_1h]
            wins = []
            t_start0 = safe_pos_float(ts[WARM], default=None) if len(ts) > WARM else None
            t_end_all = safe_pos_float(ts[-1], default=None) if ts else None
            if t_start0 is not None and t_end_all is not None and (t_end_all - t_start0) >= W90_SEC:
                s = WARM
                while s < len(ohlc):
                    t_s = safe_float(ts[s], 0.0)
                    if t_s + W90_SEC > t_end_all + 3600.0:
                        break
                    end_idx = bisect_left(ts, t_s + W90_SEC, s)
                    if end_idx - s < 12:
                        break
                    r = run(
                        None, params, peak, 60,
                        ohlc=ohlc[s - WARM:end_idx],
                        ts=ts[s - WARM:end_idx],
                    )
                    wins.append([r["ret"], r["dd"], r["bh"]])
                    s = max(s + 1, bisect_left(ts, t_s + STEP30_SEC, s + 1))
            else:
                for s in range(WARM, len(bars_1h) - W90_1H + 1, STEP_1H):
                    r = run(
                        None, params, peak, 60,
                        ohlc=ohlc[s - WARM:s + W90_1H],
                        ts=ts[s - WARM:s + W90_1H],
                    )
                    wins.append([r["ret"], r["dd"], r["bh"]])
            out["w90"] = wins
        out["secs"] = round(time.time() - t0, 1)
        lru_dict_put(_TASK_CACHE, cache_key, out, max_entries=512)
        return out
    except Exception:
        return {"profile": profile, "variant": variant, "asset": asset, "error": traceback.format_exc()}


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(EXP, exist_ok=True)
    assets = sorted({a for f in os.listdir(DATA) if (a := _extract_asset_name(f)) is not None})
    if not assets:
        print(f"[t212grid] ERROR: no CSV data files found in {DATA}. Run fetch.py first.", file=sys.stderr, flush=True)
        sys.exit(1)
    assets_set = set(assets)
    for a in assets:
        load(a, "1h")
        load(a, "1d")
    todo = []
    for profile in PROFILES:
        home = HOME[profile]
        if home in assets_set:
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
    workers = resolve_grid_workers()
    print(f"[t212grid] {len(todo)} tasks ({len(canon_todo)} unique) on {workers} workers", flush=True)
    t0 = time.time()
    done_count = 0
    valid_count = 0
    rows_out: list[dict] = []
    with Pool(workers) as pool:
        for r in pool.imap(task, canon_todo):
            p, pk = params_for(r["profile"], r["variant"])
            aliases = canon_groups.get((r["asset"], repr(p), bool(pk)), [(r["profile"], r["variant"])])
            for idx, (al_prof, al_var) in enumerate(aliases):
                done_count += 1
                row = r if idx == 0 and (al_prof, al_var) == (r["profile"], r["variant"]) else {
                    **r, "profile": al_prof, "variant": al_var, "secs": 0.0
                }
                rows_out.append(row)
                if "error" not in row and ("1h" in row or "1d" in row):
                    valid_count += 1
                err_lines = str(row.get("error") or "").strip().splitlines()
                tag = ("ERROR " + (err_lines[-1] if err_lines else "unknown")) if "error" in row else \
                    f"1h {row.get('1h', {}).get('ret')} 1d {row.get('1d', {}).get('ret')} ({row.get('secs')}s)"
                print(f"[t212grid] {done_count}/{len(todo)} {row['profile']} {row['variant']:<16} {row['asset']:<6} {tag} [{time.time()-t0:.0f}s]", flush=True)
    if valid_count == 0:
        print(
            "[t212grid] ERROR: 0 valid grid rows produced; refusing to overwrite results.jsonl",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    atomic_write_jsonl(OUT, rows_out)
