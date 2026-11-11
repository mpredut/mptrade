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
import dataclasses, os, sys, time, traceback
from datetime import datetime, timezone
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import (
    int_env,
    load_local_module,
    lru_dict_put,
    optional_env,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
EXP = optional_env("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
DATA = os.path.join(EXP, "data")
OUT = os.path.join(EXP, "results.jsonl")
from state_io import append_jsonl_record, read_jsonl_dicts
from strategies import spot_engine as strat, spot_replay as rp
from offline.backtests.datasets import load_ohlc_tuples_csv  # noqa: E402
from offline.research.kraken_continuous_grid.report import _matches_known_spec  # noqa: E402

load_isolated_kraken_cfg = load_local_module(
    os.path.join(ROOT, "kraken"), "kraken_common", module_key="kraken_common"
).load_isolated_kraken_cfg
_KRAKEN_CFG = load_isolated_kraken_cfg(os.path.join(ROOT, "kraken"))

FLAGS = {"peak": False}
_orig_step = strat.Strategy.step


def _step(self, price, *args, **kwargs):
    if FLAGS["peak"]:
        rp.ratchet_post_exit_peak_sell_price(self.s, price, qty_eps=1e-12, require_tp_kind=True)
    return _orig_step(self, price, *args, **kwargs)


R = dataclasses.replace
LIVE = strat.StratParams.from_env(_KRAKEN_CFG)
OLD = R(LIVE, takeprofit_pct=5.0, dca_spacing_growth_pct=0.0, tp_trail_adaptive=False,
        tp_regime_gate=False, tp_trail_profit_floor_pct=0.0, stop_loss_pct=12.5,
        reentry_hybrid_enabled=False, reentry_bear_bounce_pct=0.0,
        reentry_ttl_hours=0.0, reentry_pullback_adaptive=False, reentry_drop_pct=2.2,
        tp_mdhtp_enabled=False, sl_mhdsl_enabled=False,
        trend_overlay=False, dca_trend_brake=False, surge_guard=False,
        fast_profit_guard=False, tp_dynamic_flat=False)


def ov(p, topup, trail, brk=False):
    return R(p, trend_overlay=True, trend_topup=safe_nonneg_float(topup, 0.0),
             trend_trail_pct=safe_nonneg_float(trail, 0.0), trend_exit_break=brk)


CONFIGS = {
    # A. baselines
    "live": (LIVE, 0), "old_live_pre0923": (OLD, 0),
    # B. ablation of the 23 Sep promotion (live with ONE change reverted)
    "abl_tp5": (R(LIVE, takeprofit_pct=5.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 0),
    "abl_spacing0": (R(LIVE, dca_spacing_growth_pct=0.0), 0),
    "abl_trail_fixed": (R(LIVE, tp_mdhtp_enabled=False, tp_trail_adaptive=False), 0),
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
    "reentry0": (R(LIVE, reentry_drop_pct=0.0, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 0),
    "reentry1": (R(LIVE, reentry_drop_pct=1.0, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 0),
    "reentry4": (R(LIVE, reentry_drop_pct=4.0, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 0),
    "reentry_adaptive": (R(LIVE, reentry_adaptive=True, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 0),
    "peak2.2": (R(LIVE, reentry_drop_pct=2.2, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 1),
    "peak4": (R(LIVE, reentry_drop_pct=4.0, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 1),
    "peak2.2_oldlive": (OLD, 1),
    "ov650_t8_peak4": (ov(R(LIVE, reentry_drop_pct=4.0, reentry_ttl_hours=0.0, reentry_bear_bounce_pct=0.0, reentry_hybrid_enabled=False), 650, 8), 1),
    # E. stop loss
    "sl0": (R(LIVE, stop_loss_pct=0.0), 0), "sl8": (R(LIVE, stop_loss_pct=8.0), 0),
    "sl25": (R(LIVE, stop_loss_pct=25.0), 0),
    # F. take profit
    "tp3": (R(LIVE, takeprofit_pct=3.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 0),
    "tp6": (R(LIVE, takeprofit_pct=6.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 0),
    "tp8": (R(LIVE, takeprofit_pct=8.0, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 0),
    "tp_classic": (R(LIVE, tp_trend_hold=False, tp_mdhtp_enabled=False, tp_dynamic_flat=False), 0),
    # G. DCA spacing
    "dca0.75": (R(LIVE, dca_drop_pct=0.75), 0), "dca1.5": (R(LIVE, dca_drop_pct=1.5), 0),
    "dca2.5": (R(LIVE, dca_drop_pct=2.5), 0),
    "spacing0.5": (R(LIVE, dca_spacing_growth_pct=0.5), 0),
    # I. risk overlays
    "dca_brake": (R(LIVE, dca_trend_brake=True, dca_brake_min_pct=1.5), 0),
    "dca_vol_m1": (R(LIVE, dca_vol_scale_k=-1.0, dca_vol_ref=2.0), 0),
    "dca_vol_p1": (R(LIVE, dca_vol_scale_k=1.0, dca_vol_ref=2.0), 0),
}

W30, W90, WSTEP, WARM = 180, 540, 180, 42


_LOAD_CACHE: dict = {}
_ASSET_CACHE: dict = {}
_TASK_CACHE: dict = {}


def load(path):
    if isinstance(path, (bool, int, float)):
        return []
    key = os.path.abspath(os.fspath(path))
    try:
        mtime_ns = os.stat(key).st_mtime_ns
    except OSError:
        mtime_ns = None
    cached = _LOAD_CACHE.get((key, mtime_ns))
    if cached is not None:
        return cached
    rows = load_ohlc_tuples_csv(path, strict=True)
    lru_dict_put(_LOAD_CACHE, (key, mtime_ns), rows, max_entries=64)
    return rows


def load_asset(asset):
    asset_key = (os.path.abspath(DATA), asset)
    cached = _ASSET_CACHE.get(asset_key)
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
    lru_dict_put(_ASSET_CACHE, asset_key, res, max_entries=64)
    return res


def resolve_grid_workers() -> int:
    raw_w = int_env("WORKERS")
    if raw_w is not None and raw_w > 0:
        return raw_w
    return min(4, os.cpu_count() or 2)


def replay(bars, warm, params, peak, fee=0.26, initial_cash=None):
    with rp.post_exit_peak_step_scope(strat.Strategy, FLAGS, peak, _step, _orig_step):
        has_ts = bool(bars and len(bars[0]) == 5)
        has_w_ts = bool(warm and len(warm[0]) == 5)
        ohlc = [b[1:] for b in bars] if has_ts else bars
        w_ohlc = [b[1:] for b in warm] if has_w_ts else warm
        ts = [b[0] for b in bars] if has_ts else None
        w_ts = [b[0] for b in warm] if has_w_ts else None
        if not ohlc:
            if initial_cash is not None:
                return {"ret": 0.0, "dd": 0.0, "cyc": 0, "refused": 0, "min_cash": safe_nonneg_float(initial_cash, 0.0)}
            return {"ret": 0.0, "dd": 0.0, "cyc": 0, "expo": 0.0, "bh": 0.0}
        fee_val = safe_nonneg_float(fee, 0.0)
        interval_params = rp.prepare_spot_interval_params(params, 240)
        m = rp.run_replay(
            ohlc, interval_params, fee_pct=fee_val, bar_minutes=240,
            warmup_ohlc=w_ohlc, initial_cash=initial_cash,
            timestamps=ts, warmup_timestamps=w_ts,
        )
        tot = safe_float(m.get("total"), 0.0)
        if initial_cash is not None:
            ic = safe_pos_float(initial_cash, 0.0)
            funding = m.get("funding") if isinstance(m.get("funding"), dict) else {}
            min_cash_raw = funding.get("minimum_cash")
            min_cash = safe_float(min_cash_raw, default=ic) if min_cash_raw is not None else ic
            return {"ret": round(tot / ic * 100, 2) if ic > 0 else 0.0,
                    "dd": round(safe_float(m.get("max_drawdown_pct"), 0.0), 2),
                    "cyc": safe_nonneg_int(m.get("cycles"), default=0),
                    "refused": safe_nonneg_int(funding.get("refused_buys"), default=0),
                    "min_cash": min_cash}
        return rp.summarize_grid_replay_metrics(
            m,
            budget=params.allocated_budget(),
            p0=ohlc[0][3],
            p1=ohlc[-1][3],
            fee_pct=fee_val,
            bh_digits=3,
        )


def task(args):
    asset, cfg = args
    try:
        bars, ohlc, year_slices = load_asset(asset)
        params, peak = CONFIGS[cfg]
        cache_key = (os.path.abspath(DATA), asset, len(bars), bars[0][0] if bars else 0, bars[-1][0] if bars else 0,
                     repr(params), bool(peak))
        cached = _TASK_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cfg": cfg, "secs": 0.0}
        t0 = time.time()
        start_ts = bars[0][0] if bars else 0
        end_ts = bars[-1][0] if bars else 0
        res = {"asset": asset, "cfg": cfg, "bars": len(bars),
               "start_ts": start_ts, "end_ts": end_ts, "params_repr": repr(params), "peak": bool(peak),
               "full": replay(bars[WARM:], bars[:WARM], params, peak), "year": {}, "w30": [], "w90": []}
        for y_str, s_idx, e_idx in year_slices:
            res["year"][y_str] = replay(bars[s_idx:e_idx], bars[s_idx - WARM:s_idx], params, peak)
        for key, size in (("w30", W30), ("w90", W90)):
            for s in range(WARM, len(bars) - size + 1, WSTEP):
                r = replay(bars[s:s + size], bars[s - WARM:s], params, peak)
                res[key].append([r["ret"], r["dd"], r["bh"], r["cyc"]])
        res["secs"] = round(time.time() - t0, 1)
        lru_dict_put(_TASK_CACHE, cache_key, res, max_entries=512)
        return res
    except Exception:
        return {"asset": asset, "cfg": cfg, "error": traceback.format_exc()}


def _row_matches_current_spec(r: dict, assets_set: set[str]) -> bool:
    if (
        "error" in r
        or r.get("asset") not in assets_set
        or r.get("cfg") not in CONFIGS
        or not isinstance(r.get("full"), dict)
        or not isinstance(r.get("w30"), list)
        or not isinstance(r.get("w90"), list)
    ):
        return False
    bars, _, _ = load_asset(r["asset"])
    if r.get("bars") != len(bars):
        return False
    start_ts = bars[0][0] if bars else 0
    end_ts = bars[-1][0] if bars else 0
    if "start_ts" in r and r.get("start_ts") != start_ts:
        return False
    if "end_ts" in r and r.get("end_ts") != end_ts:
        return False
    return _matches_known_spec(r, CONFIGS)


if __name__ == "__main__":
    assets = sorted(f[:-9] for f in os.listdir(DATA) if f.endswith("_240m.csv"))
    assets_set = set(assets)
    for a in assets:
        load_asset(a)
    done = {
        (r["asset"], r["cfg"])
        for r in read_jsonl_dicts(OUT)
        if _row_matches_current_spec(r, assets_set)
    }
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
    workers = resolve_grid_workers()
    print(f"[grid] {len(assets)} assets x {len(CONFIGS)} configs, {len(todo)} tasks ({len(canon_todo)} unique) on {workers} workers", flush=True)
    t0 = time.time()
    done_count = 0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(task, canon_todo):
            p, pk = CONFIGS[r["cfg"]]
            aliases = canon_groups.get((r["asset"], repr(p), bool(pk)), [r["cfg"]])
            for idx, alias_cfg in enumerate(aliases):
                done_count += 1
                row = r if idx == 0 and alias_cfg == r["cfg"] else {**r, "cfg": alias_cfg, "secs": 0.0}
                append_jsonl_record(OUT, row)
                tag = "ERROR" if "error" in row else f"full {row['full']['ret']:+8.2f}% dd {row['full']['dd']:6.2f} ({row['secs']}s)"
                print(f"[grid] {done_count}/{len(todo)} {row['asset']:<9} {row['cfg']:<20} {tag}  [{time.time()-t0:.0f}s]", flush=True)
    print("[grid] done", flush=True)
