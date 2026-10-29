#!/usr/bin/env python3
"""Real 1s-cadence short-trend estimator benchmark (dense cache_365_price_*.jsonl).

Yardstick for any change to pricewindow's short-trend SSOT (growth_coefficient).
Every candidate is a causal linear filter on (optionally median-filtered) price steps,
evaluated on raw ticks exactly like the live PriceWindow (window = 222 samples at the
1.0s median cadence, recent = 60 samples). LEGACY and SW p1 match pricewindow to 1e-12.
Scored in-sample (first 60% of time) and out-of-sample (last 40%) on the decisions
the bots actually take:
  GT_S BUY  : on a dip, wait while signal says "falling" (max 90s)  -> entry savings bps
  GT_S SELL : on a rally, wait while signal says "rising" (max 90s) -> exit savings bps
  TP ride   : after a rally, hold while signal says "rising" (max 6min, 20bp trail)
plus 60s/180s forward-return spread, flips/min and turning-point lag.
With BENCH_ONLY="name1,name2,..." it also prints a paired bootstrap vs LEGACY.

Result on 27 Oct 2026 (BTC 6.78M + TAO 3.97M + ARB 0.51M ticks, 14 Jul - 27 Oct):
  - every estimator is mildly anti-predictive at 60-180s (hit 47.7-48.3%);
  - the causal median filter hurts out-of-sample in every family;
  - Step-WOLS cubic + median 5s was significantly worse than LEGACY out-of-sample
    (BTC -1.39 bps/decision, 95% CI [-2.21, -0.57]; TAO -1.40 [-2.51, -0.42]);
  - nothing beat LEGACY significantly; bot replays (monitortrades/tradeall) tied.

usage: short_trend_estimator_benchmark.py [--data-dir DIR] [SYMBOL ...]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, median_filter
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import int_env, safe_float, safe_pos_float
from cachepaths import cache_symbol_variants, get_cachedb_dir, resolve_cachedb_filepath

D = get_cachedb_dir()
N_WIN = 222
N_REC = 60
IS_FRAC = 0.6


def _set_data_dir(path):
    global D
    D = path


def load_series(sym):
    """Return (ts_sec, px) from <D>/cache_365_price_<sym>.npz, building it from cache_365_price_<sym>.jsonl."""
    npz = str(resolve_cachedb_filepath(os.path.join(D, f"cache_365_price_{sym}.npz")))
    if os.path.exists(npz):
        z = np.load(npz)
        return z["ts"], z["px"]
    ts, px = [], []
    sym_cands = cache_symbol_variants(sym)
    jsonl_path = next(
        (
            p
            for cand in sym_cands
            for prefix in ("cache_365_price_", "cache_sparse_price_")
            if os.path.exists(p := str(resolve_cachedb_filepath(os.path.join(D, f"{prefix}{cand}.jsonl"))))
        ),
        None,
    )
    if not jsonl_path:
        return np.empty(0), np.empty(0)
    from state_io import read_jsonl_dicts

    valid_syms = set(sym_cands) | {sym}
    for r in read_jsonl_dicts(jsonl_path):
        if r.get("s") not in valid_syms:
            continue
        item = r.get("i")
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t = safe_float(item[0], default=None)
        p = safe_pos_float(item[1], default=None)
        if t is not None and p is not None:
            ts.append(t / 1000.0)
            px.append(p)
    ts, px = np.asarray(ts), np.asarray(px)
    o = np.argsort(ts, kind="stable")
    ts, px = ts[o], px[o]
    keep = np.concatenate(([True], np.diff(ts) > 0))
    ts, px = ts[keep], px[keep]
    np.savez(npz, ts=ts, px=px)
    return ts, px


def conv_steps(series, w):
    """sig[t] = sum_k w_k * dy_{t-m+k}, k=1..m (w normalized), m = N_WIN-1."""
    dy = np.diff(series, prepend=series[0])
    w = w / w.sum()
    full = fftconvolve(dy, w, mode="full")[: len(dy)]
    # fftconvolve(dy, w)[t] = sum_j w_j dy_{t-j}; we need newest step weighted w[-1]
    sig = np.zeros_like(series)
    sig[N_WIN - 1:] = full[N_WIN - 1:]
    return sig


def kernel_newest_first(w_oldest_first):
    return w_oldest_first[::-1]


def legacy(series):
    m = N_WIN - 1
    k = np.arange(1, m + 1, dtype=np.float64)
    w_ols = k * (N_WIN - k)
    sf = conv_steps(series, kernel_newest_first(w_ols))
    gr = np.zeros_like(series)
    b = np.arange(N_REC, len(series))
    gr[b] = ((series[b] - series[b - 1]) + (series[b] - series[b - N_REC + 1])
             + (series[b] - series[b - N_REC])) / (2.0 * N_REC)
    return 0.5 * (sf + gr)


def step_wols(series, p):
    k = np.arange(1, N_WIN, dtype=np.float64)
    return conv_steps(series, kernel_newest_first(k ** p))


def exp_steps(series, half_life):
    k = np.arange(1, N_WIN, dtype=np.float64)
    lam = float(np.log(2.0)) / half_life
    return conv_steps(series, kernel_newest_first(np.exp(lam * (k - (N_WIN - 1)))))


def rolling_eps(px, win=N_WIN):
    dy = np.diff(px, prepend=px[0])
    kernel = np.ones(win, dtype=np.float64) / win
    mean_g = fftconvolve(dy, kernel, mode="full")[:len(px)]
    var_g = np.maximum(0.0, fftconvolve(dy**2, kernel, mode="full")[:len(px)] - mean_g**2)
    return np.sqrt(var_g)


def noise_gate(sig, px, win=N_WIN, k=0.25, eps=None):
    """Zero-out trend signals that do not exceed k * rolling_std(gradient)."""
    if eps is None:
        eps = rolling_eps(px, win=win)
    return np.where(np.abs(sig) > k * eps, sig, 0.0)


def platt_scale(sig, thr=0.6):
    """Parametric sigmoid confidence gating: maps normalized slope to probability."""
    s_std = np.std(sig)
    if s_std <= 1e-12:
        return sig
    z = sig / s_std
    prob = 1.0 / (1.0 + np.exp(-z))
    return np.where(prob >= thr, sig, np.where(prob <= 1.0 - thr, sig, 0.0))


def build_candidates(px):
    """Yield (name, signal) lazily to keep memory bounded on 6.8M-tick series."""
    only = set(filter(None, os.environ.get("BENCH_ONLY", "").split(",")))
    want = (lambda n: (not only) or n in only)
    yield "LEGACY (sf+gr)/2", legacy(px)
    for p in (1, 2, 3, 4, 6, 8):
        if want(f"SW p{p}"):
            yield f"SW p{p}", step_wols(px, p)
    for hl in (15, 30, 60):
        if want(f"EXP hl{hl}"):
            yield f"EXP hl{hl}", exp_steps(px, hl)

    eps_val = None
    for p in (2, 3):
        base_sig = None
        for k_eps in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10, 0.25):
            cname = f"SW p{p} + NoiseGate{k_eps}"
            if want(cname):
                if base_sig is None:
                    base_sig = step_wols(px, p)
                if eps_val is None:
                    eps_val = rolling_eps(px)
                yield cname, noise_gate(base_sig, px, k=k_eps, eps=eps_val)
        for p_thr in (0.55, 0.60, 0.65):
            cname = f"SW p{p} + Platt{p_thr}"
            if want(cname):
                if base_sig is None:
                    base_sig = step_wols(px, p)
                yield cname, platt_scale(base_sig, thr=p_thr)


def fwd_index(ts, h):
    idx = np.searchsorted(ts, ts + h)
    ok = idx < len(ts)
    idx = np.minimum(idx, len(ts) - 1)
    ok &= (ts[idx] - ts - h) <= 5.0
    return idx, ok


def turning_points(ts, px, half=180, prom=0.003):
    smooth = median_filter(px, size=61, mode="nearest")
    mx = maximum_filter1d(smooth, size=2 * half + 1, mode="nearest")
    mn = minimum_filter1d(smooth, size=2 * half + 1, mode="nearest")
    is_max = (smooth == mx) & ((smooth - mn) / smooth >= prom)
    is_min = (smooth == mn) & ((mx - smooth) / smooth >= prom)
    def dedup(ix):
        out, last = [], -10 ** 9
        for i in ix:
            if i - last >= half:
                out.append(int(i)); last = int(i)
        return out
    return dedup(np.where(is_min)[0]), dedup(np.where(is_max)[0])


def trigger_indices(ts, px, lo, hi, lookback_s, thr_bps, sign, min_gap_s):
    j, ok = fwd_index(ts, -lookback_s)  # index ~lookback ago (searchsorted on negative h)
    j = np.searchsorted(ts, ts - lookback_s)
    move = (px - px[j]) / px[j] * 1e4
    cand = np.where((move * sign >= thr_bps) & (np.arange(len(ts)) >= lo) & (np.arange(len(ts)) < hi))[0]
    out, last_t = [], -1e18
    for i in cand:
        if ts[i] - last_t >= min_gap_s:
            out.append(int(i)); last_t = ts[i]
    return out


def evaluate(sym):
    sys.stderr.write(f"[bench] loading {sym}...\n")
    sys.stderr.flush()
    ts, px = load_series(sym)
    N = len(ts)
    sys.stderr.write(f"[bench] evaluating {sym} ({N:,} ticks)...\n")
    sys.stderr.flush()
    t_split = ts[0] + IS_FRAC * (ts[-1] - ts[0])
    i_split = int(np.searchsorted(ts, t_split))

    i60, ok60 = fwd_index(ts, 60.0)
    i180, ok180 = fwd_index(ts, 180.0)
    r60 = (px[i60] - px) / px * 1e4
    r180 = (px[i180] - px) / px * 1e4
    mins, maxs = turning_points(ts, px)

    segs = {"IS": (N_WIN + 10, i_split), "OOS": (i_split, N - 10)}
    trig = {}
    for name, (lo, hi) in segs.items():
        trig[name] = {
            "dip": trigger_indices(ts, px, lo, hi, 120.0, 20.0, -1, 600.0),
            "rally": trigger_indices(ts, px, lo, hi, 120.0, 20.0, +1, 600.0),
        }

    def wait_exec(sig, i, want_positive, max_wait=90.0):
        t0 = ts[i]
        j = i
        while j < N - 1 and ts[j] - t0 < max_wait:
            if (sig[j] >= 0) if want_positive else (sig[j] <= 0):
                break
            j += 1
        return j

    res = {}
    for cname, sig in build_candidates(px):
        out = {}
        sgn = np.sign(sig)
        for sname, (lo, hi) in segs.items():
            sl = slice(lo, hi)
            s = sgn[sl]
            m60 = ok60[sl]; m180 = ok180[sl]
            up, dn = s > 0, s < 0
            spread60 = np.mean(r60[sl][up & m60]) - np.mean(r60[sl][dn & m60])
            spread180 = np.mean(r180[sl][up & m180]) - np.mean(r180[sl][dn & m180])
            big = m180 & (np.abs(r180[sl]) >= 8.0) & (s != 0)
            hit = np.mean(np.sign(r180[sl][big]) == s[big]) * 100
            minutes = (ts[hi - 1] - ts[lo]) / 60.0
            flips = np.sum(s[1:] * s[:-1] < 0) / minutes
            lags = []
            for ix, want in [(i, 1) for i in mins] + [(i, -1) for i in maxs]:
                if ix < lo or ix >= hi:
                    continue
                t0 = ts[ix]
                j = ix
                while j < N - 1 and ts[j] - t0 < 240 and sgn[j] != want:
                    j += 1
                lags.append(min(240.0, ts[j] - t0))
            # GT_S BUY on dips: wait while falling (sig < 0)
            buy_sav = [(px[i] - px[wait_exec(sig, i, True)]) / px[i] * 1e4 for i in trig[sname]["dip"]]
            # GT_S SELL on rallies: wait while rising (sig > 0)
            sell_sav = [(px[wait_exec(sig, i, False)] - px[i]) / px[i] * 1e4 for i in trig[sname]["rally"]]
            # TP ride after rally: hold while rising, 20bp trail, max 360s
            ride = []
            for i in trig[sname]["rally"]:
                t0, p0, peak, j = ts[i], px[i], px[i], i
                while j < N - 1 and ts[j] - t0 < 360:
                    peak = max(peak, px[j])
                    if sig[j] <= 0 or (peak - px[j]) / peak * 1e4 >= 20.0:
                        break
                    j += 1
                ride.append((px[j] - p0) / p0 * 1e4)
            out[sname] = dict(
                sp60=spread60, sp180=spread180, hit=hit, flips=flips,
                lag_med=float(np.median(lags)) if lags else np.nan,
                lag_mean=float(np.mean(lags)) if lags else np.nan,
                buy=float(np.mean(buy_sav)), sell=float(np.mean(sell_sav)), ride=float(np.mean(ride)),
                n_dip=len(buy_sav), n_rally=len(sell_sav), n_tp=len(lags),
            )
            out[sname]["edge"] = out[sname]["buy"] + out[sname]["sell"] + out[sname]["ride"]
            out[sname]["raw"] = (np.asarray(buy_sav), np.asarray(sell_sav), np.asarray(ride))
        res[cname] = out
    return sym, N, res


def main():
    global D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=D, help="directory with cache_365_price_<SYM>.jsonl (default: cachedb/)")
    ap.add_argument("symbols", nargs="*", default=["BTCUSDC", "TAOUSDC", "ARBUSDC"])
    args = ap.parse_args()
    D = args.data_dir
    os.environ["SHORT_TREND_BENCH_DATA_DIR"] = D
    syms = args.symbols
    n_workers = min(len(syms), max(1, int_env("BENCH_WORKERS") or 1))
    if n_workers <= 1:
        _set_data_dir(D)
        results = [evaluate(s) for s in syms]
    else:
        with Pool(n_workers, initializer=_set_data_dir, initargs=(D,)) as pool:
            results = pool.map(evaluate, syms)
    agg = {}
    for sym, N, res in results:
        any_c = next(iter(res.values()))
        print(f"\n=== {sym}: {N:,} ticks | IS dips/rallies/TPs={any_c['IS']['n_dip']}/{any_c['IS']['n_rally']}/{any_c['IS']['n_tp']}"
              f" | OOS={any_c['OOS']['n_dip']}/{any_c['OOS']['n_rally']}/{any_c['OOS']['n_tp']} ===")
        for c, o in res.items():
            for s in ("IS", "OOS"):
                a = agg.setdefault(c, {}).setdefault(s, {})
                for k, v in o[s].items():
                    if k == "raw":
                        continue
                    a.setdefault(k, []).append(v)
    base = "LEGACY (sf+gr)/2"
    rng = np.random.default_rng(0)
    boot_cands = list(results[0][2].keys()) if os.environ.get("BENCH_ONLY") else [
        "SW p1", "SW p2", "SW p3", "EXP hl30",
        "SW p2 + NoiseGate0.02", "SW p2 + NoiseGate0.025", "SW p2 + NoiseGate0.03", "SW p2 + NoiseGate0.04", "SW p2 + NoiseGate0.05",
        "SW p3 + NoiseGate0.02", "SW p3 + NoiseGate0.025", "SW p3 + NoiseGate0.03", "SW p3 + NoiseGate0.04", "SW p3 + NoiseGate0.05",
        "SW p3 + Platt0.6", "SW p3 + Platt0.65",
    ]
    print("\n>>> PAIRED BOOTSTRAP vs LEGACY: edge diff (bps/decision), 95% CI, per symbol & period <<<")
    for sym, N, res in results:
        for c in boot_cands:
            if c == base or c not in res or base not in res:
                continue
            cells = []
            for s in ("IS", "OOS"):
                comps = []
                for ci in range(3):
                    d = res[c][s]["raw"][ci] - res[base][s]["raw"][ci]
                    comps.append(d)
                point = sum(float(np.mean(d)) for d in comps)
                boots = np.zeros(2000)
                for d in comps:
                    idx = rng.integers(0, len(d), size=(2000, len(d)))
                    boots += d[idx].mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                cells.append(f"{s} {point:+.2f} [{lo:+.2f},{hi:+.2f}]")
            print(f"{sym:<8} {c:<24} " + " | ".join(cells))
    hdr = (f"{'Candidate':<24}|{'edgeIS':>7}{'edgeOOS':>8}|{'buyOOS':>7}{'sellOOS':>8}{'rideOOS':>8}|"
           f"{'sp60':>6}{'sp180':>7}{'hit%':>6}|{'flip/m':>7}{'lagMed':>7}{'lagAvg':>7}")
    print("\n>>> MEAN ACROSS SYMBOLS, sorted by OUT-OF-SAMPLE edge (bps per decision) <<<")
    print(hdr); print("-" * len(hdr))
    rows = []
    for c, a in agg.items():
        m = {s: {k: float(np.nanmean(v)) for k, v in a[s].items()} for s in a}
        rows.append((m["OOS"]["edge"], c, m))
    for _, c, m in sorted(rows, reverse=True):
        o = m["OOS"]
        print(f"{c:<24}|{m['IS']['edge']:>+7.2f}{o['edge']:>+8.2f}|{o['buy']:>+7.2f}{o['sell']:>+8.2f}{o['ride']:>+8.2f}|"
              f"{o['sp60']:>+6.2f}{o['sp180']:>+7.2f}{o['hit']:>6.1f}|{o['flips']:>7.2f}{o['lag_med']:>7.0f}{o['lag_mean']:>7.0f}")
    sys.stdout.flush()
    # per-symbol OOS for top candidates
    top = [c for _, c, _ in sorted(rows, reverse=True)[:8]] + ["LEGACY (sf+gr)/2"]
    print("\n>>> PER-SYMBOL OOS edge (bps) for top-8 + legacy <<<")
    for sym, N, res in results:
        print(sym, " | ".join(f"{c}: {res[c]['OOS']['edge']:+.2f}" for c in dict.fromkeys(top) if c in res))


if __name__ == "__main__":
    main()
