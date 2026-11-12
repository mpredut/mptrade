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
import io
import math
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, median_filter
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import (
    int_env,
    normalize_epoch_sec,
    optional_env,
    parse_epoch_price_pair,
    safe_float,
    safe_int,
    safe_pos_float,
    split_csv,
    temporary_environ,
)
from cachepaths import cache_symbol_variants, get_cachedb_dir, resolve_cachedb_filepath
from state_io import atomic_write_bytes, read_jsonl_dicts
from strategies.replay_metrics import validate_bt_cli_token

D = get_cachedb_dir()
N_WIN = 222
N_REC = 60
IS_FRAC = 0.6


def _set_data_dir(path):
    global D
    D = path


def _extract_valid_npz_arrays(z) -> tuple[np.ndarray, np.ndarray] | None:
    raw_ts = z["ts"]
    raw_px = z["px"]
    if raw_ts.ndim != 1 or raw_px.ndim != 1 or len(raw_ts) != len(raw_px) or len(raw_ts) == 0:
        return None
    ts_arr = np.asarray(raw_ts, dtype=np.float64)
    px_arr = np.asarray(raw_px, dtype=np.float64)
    mask = np.isfinite(ts_arr) & (ts_arr >= 0.0) & np.isfinite(px_arr) & (px_arr > 0.0)
    if not np.any(mask):
        return None
    if not np.all(mask):
        ts_arr = ts_arr[mask]
        px_arr = px_arr[mask]
    if len(ts_arr) > 1 and np.any(np.diff(ts_arr) <= 0.0):
        order = np.argsort(ts_arr, kind="stable")
        ts_arr = ts_arr[order]
        px_arr = px_arr[order]
        keep = np.concatenate(([True], np.diff(ts_arr) > 0.0))
        ts_arr = ts_arr[keep]
        px_arr = px_arr[keep]
    return ts_arr, px_arr


def load_series(sym):
    """Return (ts_sec, px) from <D>/cache_365_price_<sym>.npz, building it from cache_365_price_<sym>.jsonl."""
    try:
        sym = validate_bt_cli_token(sym, "symbol").upper()
    except ValueError:
        return np.empty(0), np.empty(0)
    npz_365 = str(resolve_cachedb_filepath(os.path.join(D, f"cache_365_price_{sym}.npz")))
    npz_sparse = str(resolve_cachedb_filepath(os.path.join(D, f"cache_sparse_price_{sym}.npz")))
    ts, px = [], []
    sym_cands = cache_symbol_variants(sym)
    chosen_prefix = None
    jsonl_path = None
    for prefix in ("cache_365_price_", "cache_sparse_price_"):
        for cand in sym_cands:
            p = str(resolve_cachedb_filepath(os.path.join(D, f"{prefix}{cand}.jsonl")))
            if os.path.exists(p):
                chosen_prefix = prefix
                jsonl_path = p
                break
        if jsonl_path:
            break
    if not jsonl_path:
        for existing_npz in (npz_365, npz_sparse):
            if os.path.exists(existing_npz):
                try:
                    with np.load(existing_npz) as z:
                        valid = _extract_valid_npz_arrays(z)
                        if valid is not None:
                            return valid
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    continue
        return np.empty(0), np.empty(0)
    npz_target = npz_365 if chosen_prefix == "cache_365_price_" else npz_sparse
    if os.path.exists(npz_target):
        try:
            if os.path.getmtime(npz_target) >= os.path.getmtime(jsonl_path):
                with np.load(npz_target) as z:
                    valid = _extract_valid_npz_arrays(z)
                    if valid is not None:
                        return valid
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass

    valid_syms = set(sym_cands) | {sym}
    for r in read_jsonl_dicts(jsonl_path):
        if r.get("s") not in valid_syms:
            continue
        if (parsed := parse_epoch_price_pair(r.get("i"))) is not None:
            ts.append(parsed[0])
            px.append(parsed[1])
    if not ts:
        return np.empty(0), np.empty(0)
    ts, px = np.asarray(ts, dtype=np.float64), np.asarray(px, dtype=np.float64)
    o = np.argsort(ts, kind="stable")
    ts, px = ts[o], px[o]
    keep = np.concatenate(([True], np.diff(ts) > 0))
    ts, px = ts[keep], px[keep]
    buf = io.BytesIO()
    np.savez(buf, ts=ts, px=px)
    atomic_write_bytes(npz_target, buf.getvalue())
    return ts, px


def conv_steps(series, w):
    """sig[t] = sum_k w_k * dy_{t-m+k}, k=1..m (w normalized), m = N_WIN-1."""
    if len(series) == 0:
        return np.empty(0, dtype=np.float64)
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
    lam = math.log(2.0) / half_life
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


def platt_scale(sig, thr=0.6, fit_end: int | None = None):
    """Parametric sigmoid confidence gating: maps normalized slope to probability (fit strictly on IS)."""
    fit_slice = sig[:fit_end] if (fit_end is not None and fit_end > 0) else sig
    s_std = np.std(fit_slice) if len(fit_slice) > 0 else 0.0
    if s_std <= 1e-12:
        return sig
    z = np.clip(sig / s_std, -60.0, 60.0)
    prob = 1.0 / (1.0 + np.exp(-z))
    return np.where(prob >= thr, sig, np.where(prob <= 1.0 - thr, sig, 0.0))


def build_candidates(px, fit_end: int | None = None):
    """Yield (name, signal) lazily to keep memory bounded on 6.8M-tick series."""
    only = set(split_csv(optional_env("BENCH_ONLY")))
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
                yield cname, platt_scale(base_sig, thr=p_thr, fit_end=fit_end)


def fwd_index(ts, h):
    if len(ts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    idx = np.searchsorted(ts, ts + h)
    ok = idx < len(ts)
    idx = np.minimum(idx, len(ts) - 1)
    ok &= (ts[idx] - ts - h) <= 5.0
    return idx, ok


def turning_points(ts, px, half=180, prom=0.003):
    if len(px) == 0:
        return [], []
    smooth = median_filter(px, size=61, mode="nearest")
    mx = maximum_filter1d(smooth, size=2 * half + 1, mode="nearest")
    mn = minimum_filter1d(smooth, size=2 * half + 1, mode="nearest")
    is_max = (smooth == mx) & ((smooth - mn) / smooth >= prom)
    is_min = (smooth == mn) & ((mx - smooth) / smooth >= prom)
    def dedup(ix):
        out, last = [], -10 ** 9
        for i in ix:
            ii = safe_int(i.item() if hasattr(i, "item") else i, 0)
            if ii - last >= half:
                out.append(ii)
                last = ii
        return out
    return dedup(np.where(is_min)[0]), dedup(np.where(is_max)[0])


def trigger_indices(ts, px, lo, hi, lookback_s, thr_bps, sign, min_gap_s):
    if len(ts) == 0 or hi <= lo:
        return []
    j = np.minimum(np.searchsorted(ts, ts - lookback_s), len(ts) - 1)
    base_px = px[j]
    move = np.divide(px - base_px, base_px, out=np.zeros_like(px, dtype=np.float64), where=base_px > 0) * 1e4
    idx_arr = np.arange(len(ts))
    cand = np.where((move * sign >= thr_bps) & (idx_arr >= lo) & (idx_arr < hi))[0]
    out, last_t = [], -1e18
    for i in cand:
        ii = safe_int(i.item() if hasattr(i, "item") else i, 0)
        if ts[ii] - last_t >= min_gap_s:
            out.append(ii)
            last_t = ts[ii]
    return out


def _safe_mean(arr) -> float:
    return safe_float(np.mean(arr).item(), default=math.nan) if len(arr) > 0 else math.nan


def _safe_nanmean(vals) -> float:
    finite = [fv for x in vals if (fv := safe_float(x, default=None)) is not None]
    return safe_float(np.mean(finite).item(), default=math.nan) if finite else math.nan


def evaluate(sym):
    sys.stderr.write(f"[bench] loading {sym}...\n")
    sys.stderr.flush()
    ts, px = load_series(sym)
    N = len(ts)
    sys.stderr.write(f"[bench] evaluating {sym} ({N:,} ticks)...\n")
    sys.stderr.flush()
    if N < N_WIN + 20:
        return sym, N, {}
    t_split = ts[0] + IS_FRAC * (ts[-1] - ts[0])
    i_split = safe_int(np.searchsorted(ts, t_split).item(), N_WIN + 11)
    i_split = min(max(N_WIN + 11, i_split), N - 11)

    i60, ok60 = fwd_index(ts, 60.0)
    i180, ok180 = fwd_index(ts, 180.0)
    r60 = (px[i60] - px) / px * 1e4
    r180 = (px[i180] - px) / px * 1e4
    mins_is, maxs_is = turning_points(ts[:i_split], px[:i_split])
    mins_oos_rel, maxs_oos_rel = turning_points(ts[i_split:], px[i_split:])
    tp_by_seg = {
        "IS": (mins_is, maxs_is),
        "OOS": ([i + i_split for i in mins_oos_rel], [i + i_split for i in maxs_oos_rel]),
    }

    segs = {"IS": (N_WIN + 10, i_split), "OOS": (i_split, N - 10)}
    trig = {}
    for name, (lo, hi) in segs.items():
        seg_lo = 0 if name == "IS" else i_split
        j_lb = np.maximum(seg_lo, np.minimum(np.searchsorted(ts, ts - 120.0), len(ts) - 1))
        base_px = px[j_lb]
        move = np.divide(px - base_px, base_px, out=np.zeros_like(px, dtype=np.float64), where=base_px > 0) * 1e4
        idx_arr = np.arange(len(ts))
        trig[name] = {}
        for key_name, sign in (("dip", -1), ("rally", +1)):
            cand = np.where((move * sign >= 20.0) & (idx_arr >= lo) & (idx_arr < hi) & (ts - ts[j_lb] >= 60.0))[0]
            sel, last_t = [], -1e18
            for i in cand:
                ii = safe_int(i.item() if hasattr(i, "item") else i, 0)
                if ts[ii] - last_t >= 600.0:
                    sel.append(ii)
                    last_t = ts[ii]
            trig[name][key_name] = sel

    def wait_exec(sig, i, want_positive, upper, max_wait=90.0):
        t0 = ts[i]
        j = i
        while j < upper - 1 and ts[j] - t0 < max_wait:
            if (sig[j] >= 0) if want_positive else (sig[j] <= 0):
                break
            if ts[j + 1] - t0 > max_wait + 5.0:
                break
            j += 1
        return j

    res = {}
    for cname, sig in build_candidates(px, fit_end=i_split):
        out = {}
        sgn = np.sign(sig)
        for sname, (lo, hi) in segs.items():
            seg_bound = i_split if sname == "IS" else N
            sl = slice(lo, hi)
            s = sgn[sl]
            m60 = ok60[sl] & (i60[sl] < seg_bound)
            m180 = ok180[sl] & (i180[sl] < seg_bound)
            up, dn = s > 0, s < 0
            spread60 = _safe_mean(r60[sl][up & m60]) - _safe_mean(r60[sl][dn & m60])
            spread180 = _safe_mean(r180[sl][up & m180]) - _safe_mean(r180[sl][dn & m180])
            big = m180 & (np.abs(r180[sl]) >= 8.0) & (s != 0)
            hit = _safe_mean(np.sign(r180[sl][big]) == s[big]) * 100
            minutes = max((ts[hi - 1] - ts[lo]) / 60.0, 1e-9) if hi > lo else 1e-9
            flips = safe_float(np.sum(s[1:] * s[:-1] < 0).item(), 0.0) / minutes
            seg_mins, seg_maxs = tp_by_seg[sname]
            lags = []
            for ix, want in [(i, 1) for i in seg_mins] + [(i, -1) for i in seg_maxs]:
                if ix < lo or ix >= hi:
                    continue
                t0 = ts[ix]
                j = ix
                while j < seg_bound - 1 and ts[j] - t0 < 240 and sgn[j] != want:
                    if ts[j + 1] - t0 > 245.0:
                        break
                    j += 1
                lags.append(min(240.0, ts[j] - t0))
            # GT_S BUY on dips: wait while falling (sig < 0)
            buy_sav = [(px[i] - px[wait_exec(sig, i, True, seg_bound)]) / px[i] * 1e4 for i in trig[sname]["dip"]]
            # GT_S SELL on rallies: wait while rising (sig > 0)
            sell_sav = [(px[wait_exec(sig, i, False, seg_bound)] - px[i]) / px[i] * 1e4 for i in trig[sname]["rally"]]
            # TP ride after rally: hold while rising, 20bp trail, max 360s
            ride = []
            for i in trig[sname]["rally"]:
                t0, p0, peak, j = ts[i], px[i], px[i], i
                while j < seg_bound - 1 and ts[j] - t0 < 360:
                    peak = max(peak, px[j])
                    if sig[j] <= 0 or (peak - px[j]) / peak * 1e4 >= 20.0:
                        break
                    if ts[j + 1] - t0 > 365.0:
                        break
                    j += 1
                ride.append((px[j] - p0) / p0 * 1e4)
            buy_m = _safe_mean(buy_sav)
            sell_m = _safe_mean(sell_sav)
            ride_m = _safe_mean(ride)
            finite_parts = [x for x in (buy_m, sell_m, ride_m) if math.isfinite(x)]
            out[sname] = dict(
                sp60=spread60, sp180=spread180, hit=hit, flips=flips,
                lag_med=safe_float(np.median(lags).item(), default=math.nan) if lags else math.nan,
                lag_mean=safe_float(np.mean(lags).item(), default=math.nan) if lags else math.nan,
                buy=buy_m, sell=sell_m, ride=ride_m,
                n_dip=len(buy_sav), n_rally=len(sell_sav), n_tp=len(lags),
            )
            out[sname]["edge"] = sum(finite_parts) if finite_parts else math.nan
            out[sname]["raw"] = (np.asarray(buy_sav, dtype=np.float64), np.asarray(sell_sav, dtype=np.float64), np.asarray(ride, dtype=np.float64))
        res[cname] = out
    return sym, N, res


def main():
    global D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=D, help="directory with cache_365_price_<SYM>.jsonl (default: cachedb/)")
    ap.add_argument("symbols", nargs="*", default=["BTCUSDC", "TAOUSDC", "ARBUSDC"])
    args = ap.parse_args()
    syms: list[str] = []
    for raw_sym in args.symbols:
        try:
            syms.append(validate_bt_cli_token(raw_sym, "symbol").upper())
        except ValueError as exc:
            ap.error(str(exc))
    if not syms:
        ap.error("at least one symbol is required")
    D = args.data_dir
    with temporary_environ(SHORT_TREND_BENCH_DATA_DIR=D):
        n_workers = min(len(syms), max(1, int_env("BENCH_WORKERS") or 1))
        if n_workers <= 1:
            _set_data_dir(D)
            results = [evaluate(s) for s in syms]
        else:
            with Pool(n_workers, initializer=_set_data_dir, initargs=(D,)) as pool:
                results = pool.map(evaluate, syms)
    agg = {}
    for sym, N, res in results:
        if not res:
            print(f"\n=== {sym}: {N:,} ticks (insufficient history) ===")
            continue
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
    if not agg:
        sys.stderr.write("[bench] ERROR: no symbols had sufficient tick history for evaluation\n")
        sys.stderr.flush()
        sys.exit(1)
    base = "LEGACY (sf+gr)/2"
    rng = np.random.default_rng(0)
    non_empty_res = [r[2] for r in results if r[2]]
    boot_cands = (list(non_empty_res[0].keys()) if non_empty_res else []) if optional_env("BENCH_ONLY") else [
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
                d_buy = res[c][s]["raw"][0] - res[base][s]["raw"][0]
                d_sell = res[c][s]["raw"][1] - res[base][s]["raw"][1]
                d_ride = res[c][s]["raw"][2] - res[base][s]["raw"][2]
                finite_d = [x for x in (_safe_mean(d_buy), _safe_mean(d_sell), _safe_mean(d_ride)) if math.isfinite(x)]
                point = sum(finite_d) if finite_d else math.nan
                boots = np.zeros(2000)
                has_any_boot = False
                if len(d_buy) > 0:
                    idx_b = rng.integers(0, len(d_buy), size=(2000, len(d_buy)))
                    boots += d_buy[idx_b].mean(axis=1)
                    has_any_boot = True
                # Pair sell_sav and ride on their shared rally trigger indices to preserve covariance
                if len(d_sell) > 0 and len(d_sell) == len(d_ride):
                    d_rally = d_sell + d_ride
                    idx_r = rng.integers(0, len(d_rally), size=(2000, len(d_rally)))
                    boots += d_rally[idx_r].mean(axis=1)
                    has_any_boot = True
                if not has_any_boot:
                    boots[:] = np.nan
                lo, hi = np.percentile(boots, [2.5, 97.5])
                cells.append(f"{s} {point:+.2f} [{lo:+.2f},{hi:+.2f}]")
            print(f"{sym:<8} {c:<24} " + " | ".join(cells))
    hdr = (f"{'Candidate':<24}|{'edgeIS':>7}{'edgeOOS':>8}|{'buyOOS':>7}{'sellOOS':>8}{'rideOOS':>8}|"
           f"{'sp60':>6}{'sp180':>7}{'hit%':>6}|{'flip/m':>7}{'lagMed':>7}{'lagAvg':>7}")
    print("\n>>> MEAN ACROSS SYMBOLS, sorted by OUT-OF-SAMPLE edge (bps per decision) <<<")
    print(hdr); print("-" * len(hdr))
    rows = []
    for c, a in agg.items():
        m = {s: {k: _safe_nanmean(v) for k, v in a[s].items()} for s in a}
        rows.append((m["OOS"]["edge"], c, m))
    row_sort_key = lambda r: (math.isnan(r[0]), -r[0] if not math.isnan(r[0]) else 0.0, r[1])
    sorted_rows = sorted(rows, key=row_sort_key)
    for _, c, m in sorted_rows:
        o = m["OOS"]
        print(f"{c:<24}|{m['IS']['edge']:>+7.2f}{o['edge']:>+8.2f}|{o['buy']:>+7.2f}{o['sell']:>+8.2f}{o['ride']:>+8.2f}|"
              f"{o['sp60']:>+6.2f}{o['sp180']:>+7.2f}{o['hit']:>6.1f}|{o['flips']:>7.2f}{o['lag_med']:>7.0f}{o['lag_mean']:>7.0f}")
    sys.stdout.flush()
    # per-symbol OOS for top candidates
    top = [c for _, c, _ in sorted_rows[:8]] + ["LEGACY (sf+gr)/2"]
    print("\n>>> PER-SYMBOL OOS edge (bps) for top-8 + legacy <<<")
    for sym, N, res in results:
        print(sym, " | ".join(f"{c}: {res[c]['OOS']['edge']:+.2f}" for c in dict.fromkeys(top) if c in res))


if __name__ == "__main__":
    main()
