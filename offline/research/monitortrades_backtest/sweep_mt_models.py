#!/usr/bin/env python3
"""Fast vectorized multi-model portfolio replay backtest for monitortrades.py.

Evaluates Legacy, uniform Step-WOLS (p=2, p=3), Adaptive Noise-Gating (k=0.0..0.25
and scale-invariant z-score gating), and the Differentiated Per-Coin Step-WOLS
architecture (BTC p=3.0, Altcoins p=2.0 from instruments.conf) across all
monitortrades instruments (BTCUSDC, TAOUSDC, ARBUSDC, HYPEUSD) on:
  1. Full 329-day sparse archive (cache_sparse_price_*.jsonl)
  2. Dense 1-second archive (cache_365_price_*.jsonl / *.npz)
"""
from __future__ import annotations

import contextlib
import gc
import inspect
import os
import sys
import time
import zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from botcore import _parse_bool_flag, exact_pos_int, safe_float, safe_int, safe_pos_float
import pricewindow as pw
from cachepaths import cache_path
from providers.replay_provider import load_price_series
import offline.research.monitortrades_backtest.run_replay_backtest as rrb


def compute_window_bounds(ts: np.ndarray, win_sec: float = rrb.DEFAULT_WINDOW_SECONDS):
    """Replicate exact cur_len, sample_rate, and [a, b] bounds of prewarm_trend_series."""
    w_sec = safe_pos_float(win_sec, rrb.DEFAULT_WINDOW_SECONDS)
    N = len(ts)
    a_arr = np.zeros(N, dtype=np.int64)
    sr_arr = np.full(N, pw.DEFAULT_SAMPLE_RATE_SEC, dtype=np.float64)
    prev_ts, win_size, cur_len, sample_rate = None, 200, 0, pw.DEFAULT_SAMPLE_RATE_SEC
    for i in range(N):
        t = safe_float(ts[i].item() if hasattr(ts[i], "item") else ts[i], 0.0)
        prev_ts, win_size, cur_len, sample_rate = rrb.step_adaptive_trend_window(
            t,
            prev_ts,
            win_size,
            cur_len,
            sample_rate,
            w_sec,
            default_win_size=10,
        )
        a_arr[i] = i - cur_len + 1
        sr_arr[i] = sample_rate
    return a_arr, sr_arr


def compute_window_signals_from_bounds(
    px: np.ndarray,
    a: np.ndarray,
    sr: np.ndarray,
    *,
    symbol: str | None = None,
    recent_win_sec: float | None = None,
    min_valid_n: int = 2,
) -> dict[str, np.ndarray]:
    """Vectorized chunked O(1) prefix-sum evaluation of short-trend estimators given window bounds [a, b]."""
    N = len(px)
    if N == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return {
            "n": empty_i,
            "m": empty_i,
            "eps": empty_f,
            "scale_inv": empty_f,
            "gr": empty_f,
            "sf": empty_f,
            "legacy_raw": empty_f,
            "legacy": empty_f,
            "sw1": empty_f,
            "sw2": empty_f,
            "sw3": empty_f,
            "diff": empty_f,
            "gc": empty_f,
        }

    b = np.arange(N, dtype=np.int64)
    n = b - a + 1
    m = b - a
    valid = n >= min_valid_n
    n_f = np.maximum(1.0, n.astype(np.float64))
    m_f = np.maximum(1.0, m.astype(np.float64))

    recent_sec = pw.effective_recent_gradient_seconds(recent_win_sec)
    r_arr = np.maximum(2, (recent_sec / np.maximum(1e-9, sr)).astype(np.int64))
    b_m1 = np.maximum(0, b - 1)
    a_p1 = np.minimum(N - 1, a + 1)
    sum_g_all = (px[b] - px[a]) + 0.5 * ((px[b] - px[b_m1]) + (px[a_p1] - px[a]))

    b_mr = np.maximum(0, b - r_arr)
    b_mr1 = np.maximum(0, b - r_arr + 1)
    gr_sub = ((px[b] - px[b_m1]) + (px[b] - px[b_mr1]) + (px[b] - px[b_mr])) / (2.0 * r_arr.astype(np.float64))
    gr_all = sum_g_all / n_f
    gradient_recent = np.where(n > r_arr, gr_sub, gr_all)
    gradient_recent = np.where(np.abs(gradient_recent) <= 1e-12, 0.0, gradient_recent)

    g_int = np.zeros(N, dtype=np.float64)
    if N >= 3:
        g_int[1:-1] = 0.5 * (px[2:] - px[:-2])

    slope_full = np.zeros(N, dtype=np.float64)
    gc_sw1 = np.zeros(N, dtype=np.float64)
    gc_sw2 = np.zeros(N, dtype=np.float64)
    gc_sw3 = np.zeros(N, dtype=np.float64)
    eps = np.zeros(N, dtype=np.float64)

    # Chunked prefix sums prevent float64 precision loss when k^2 * dy exceeds 2^53 on long series
    CHUNK = 4096
    for hi_start in range(0, N, CHUNK):
        hi_end = min(N, hi_start + CHUNK)
        lo_start = safe_int(np.min(a[hi_start:hi_end]).item(), 0)
        px_sub = px[lo_start:hi_end]
        g_sub = g_int[lo_start:hi_end]
        L = len(px_sub)

        y_ref = px_sub[0]
        dy = px_sub - y_ref
        k_loc = np.arange(L, dtype=np.float64)

        P0 = np.cumsum(dy)
        P1 = np.cumsum(k_loc * dy)
        P2 = np.cumsum(dy * dy)
        Q2 = np.cumsum(k_loc * k_loc * dy)
        PG2 = np.cumsum(g_sub * g_sub)

        def _rsum(P, lo, hi):
            prev = np.where(lo > 0, P[np.maximum(0, lo - 1)], 0.0)
            return P[hi] - prev

        a_loc = a[hi_start:hi_end] - lo_start
        b_loc = b[hi_start:hi_end] - lo_start
        bm1_loc = np.maximum(0, b_loc - 1)
        ap1_loc = np.minimum(L - 1, a_loc + 1)
        m_sub = m_f[hi_start:hi_end]
        n_sub = n_f[hi_start:hi_end]
        val_sub = valid[hi_start:hi_end]

        s0 = _rsum(P0, a_loc, b_loc)
        s1 = _rsum(P1, a_loc, b_loc)
        s2 = _rsum(P2, a_loc, b_loc)
        ss_y = s2 - (s0 * s0) / n_sub
        cov_xy = s1 - (a_loc.astype(np.float64) + 0.5 * (n_sub - 1.0)) * s0
        denom_ols = np.maximum(1.0, n_sub * (n_sub * n_sub - 1.0))
        slope_full[hi_start:hi_end] = np.where(val_sub & (ss_y > 1e-18), (12.0 * cov_xy) / denom_ols, 0.0)

        a0 = _rsum(P0, a_loc, bm1_loc)
        a1 = _rsum(P1, a_loc, bm1_loc)
        a2 = _rsum(Q2, a_loc, bm1_loc)
        c_f = a_loc.astype(np.float64)
        dy_b = dy[b_loc]

        # p=1:
        total_p1 = m_sub * dy_b - a0
        s1_w = 0.5 * m_sub * (m_sub + 1.0)
        gc_sw1[hi_start:hi_end] = np.where(val_sub, total_p1 / s1_w, 0.0)

        # p=2:
        sum_2r1 = 2.0 * a1 - (2.0 * c_f - 1.0) * a0
        total_p2 = (m_sub * m_sub) * dy_b - sum_2r1
        s2_w = (m_sub * (m_sub + 1.0) * (2.0 * m_sub + 1.0)) / 6.0
        gc_sw2[hi_start:hi_end] = np.where(val_sub, total_p2 / s2_w, 0.0)

        # p=3:
        sum_r2 = a2 - 2.0 * c_f * a1 + c_f * c_f * a0
        sum_r1 = a1 - c_f * a0
        total_p3 = (m_sub ** 3) * dy_b - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
        s3_w = (0.5 * m_sub * (m_sub + 1.0)) ** 2
        gc_sw3[hi_start:hi_end] = np.where(val_sub, total_p3 / s3_w, 0.0)

        # Epsilon (population std of np.gradient(px[a:b+1])):
        g0 = px_sub[ap1_loc] - px_sub[a_loc]
        gn = px_sub[b_loc] - px_sub[bm1_loc]
        sum_g2 = g0 * g0 + (PG2[bm1_loc] - PG2[a_loc]) + gn * gn
        mean_g = sum_g_all[hi_start:hi_end] / n_sub
        var_g = np.maximum(0.0, sum_g2 / n_sub - mean_g * mean_g)
        eps[hi_start:hi_end] = np.where(val_sub & (n_sub >= 3), np.sqrt(var_g), 0.0)

    tol_raw = 1e-10 * np.maximum(1.0, np.abs(px[b]))
    legacy_raw = 0.5 * (slope_full + gradient_recent)
    gc_legacy = np.where(valid, legacy_raw, 0.0)
    gc_legacy = np.where(np.abs(gc_legacy) <= tol_raw, 0.0, gc_legacy)
    gc_sw1 = np.where(np.abs(gc_sw1) <= tol_raw, 0.0, gc_sw1)
    gc_sw2 = np.where(np.abs(gc_sw2) <= tol_raw, 0.0, gc_sw2)
    gc_sw3 = np.where(np.abs(gc_sw3) <= tol_raw, 0.0, gc_sw3)

    sym_p = pw.step_wols_power_for_symbol(symbol)
    gc_diff = gc_sw3 if sym_p >= 2.5 else (gc_sw2 if sym_p >= 1.5 else gc_sw1)

    scale_inv = np.where(m_f > 10.0, np.sqrt(10.0 / m_f), 1.0)

    return {
        "n": n,
        "m": m,
        "eps": eps,
        "scale_inv": scale_inv,
        "gr": gradient_recent,
        "sf": slope_full,
        "legacy_raw": legacy_raw,
        "legacy": gc_legacy,
        "sw1": gc_sw1,
        "sw2": gc_sw2,
        "sw3": gc_sw3,
        "diff": gc_diff,
        "gc": gc_diff,
    }


def compute_all_signals(
    series: list[tuple[float, float]],
    win_sec: float = rrb.DEFAULT_WINDOW_SECONDS,
    symbol: str | None = None,
    *,
    use_symbol_fast_window: bool = True,
):
    """Vectorized O(1) prefix-sum evaluation of short-trend estimators over series."""
    N = len(series)
    ts = np.fromiter((t for t, _ in series), dtype=np.float64, count=N)
    px = np.fromiter((p for _, p in series), dtype=np.float64, count=N)
    eff_win_sec = win_sec
    if (
        use_symbol_fast_window
        and symbol
        and np.isclose(win_sec, rrb.DEFAULT_WINDOW_SECONDS, rtol=0.0, atol=1e-6)
    ):
        fast_w = pw.fast_trend_window_for_symbol(symbol)
        fast_w_f = safe_pos_float(fast_w, default=None)
        if fast_w_f is not None:
            eff_win_sec = fast_w_f
    a, sr = compute_window_bounds(ts, win_sec=eff_win_sec)
    return compute_window_signals_from_bounds(
        px, a, sr, symbol=symbol, recent_win_sec=eff_win_sec, min_valid_n=2
    )


def make_tape(
    gc: np.ndarray,
    eps: np.ndarray,
    k_eps: float = 0.0,
    scale_inv: np.ndarray | None = None,
) -> tuple[bool, ...]:
    cond = gc > 0.0
    if k_eps > 0.0:
        eff_k = k_eps * scale_inv if scale_inv is not None else k_eps
        cond = np.where(eps > 0.0, gc > (eff_k * eps), cond)
    cond[:1] = False
    return cond.tolist()


def load_fresh_365_npz_arrays(sym: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(ts, px)`` float64 arrays from ``cache_365_price_<sym>.npz`` when fresh against ``.jsonl``."""
    npz_path = cache_path(f"cache_365_price_{sym}.npz")
    jsonl_365 = cache_path(f"cache_365_price_{sym}.jsonl")
    if os.path.exists(npz_path):
        try:
            npz_fresh = not os.path.exists(jsonl_365) or os.path.getmtime(npz_path) >= os.path.getmtime(jsonl_365)
            if npz_fresh:
                with np.load(npz_path, allow_pickle=False) as z:
                    if "ts" not in z or "px" not in z:
                        return None
                    ts_arr = np.asarray(z["ts"], dtype=np.float64)
                    px_arr = np.asarray(z["px"], dtype=np.float64)
                    if ts_arr.ndim != 1 or px_arr.ndim != 1 or len(ts_arr) != len(px_arr) or len(ts_arr) == 0:
                        return None
                    valid_mask = np.isfinite(ts_arr) & np.isfinite(px_arr) & (px_arr > 0.0)
                    if not np.any(valid_mask):
                        return None
                    return ts_arr[valid_mask], px_arr[valid_mask]
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            pass
    return None


def load_archive_series(sym: str, use_365: bool = False) -> list[tuple[float, float]]:
    """Load dense 1s (.npz / .jsonl) or sparse 329d price series for `sym`."""
    if use_365:
        arrs = load_fresh_365_npz_arrays(sym)
        if arrs is not None:
            return list(zip(arrs[0].tolist(), arrs[1].tolist()))
    path = rrb.resolve_archive_jsonl_path(sym, use_365=use_365)
    if not os.path.exists(path):
        return []
    return load_price_series(path, sym)


def _accepts_two_positional_args(fn) -> bool:
    """Return True when ``fn`` accepts at least two positional arguments (or ``*args``)."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        code_obj = getattr(fn, "__code__", None)
        return bool(code_obj is not None and getattr(code_obj, "co_argcount", 0) >= 2)
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    pos_count = sum(
        1
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return pos_count >= 2


def evaluate_mt_model_symbol_cell(
    sym: str,
    series: list,
    sig_input: object,
    tape_fn,
) -> tuple[float, float, str]:
    """Run ``tape_fn`` for ``sym`` over ``series`` and format the net-PnL/trade-count table cell."""
    key = rrb._series_cache_key(sym, series, rrb.DEFAULT_WINDOW_SECONDS)
    rrb._TREND_TAPE_CACHE.clear()
    rrb._TREND_TAPE_CACHE[key] = tuple(
        tape_fn(sig_input, sym) if _accepts_two_positional_args(tape_fn) else tape_fn(sig_input)
    )
    cfg = rrb.SYMBOLS[sym]
    res = rrb.run_symbol(sym, cfg["params"], cfg["base"], quiet=True, series=series) or {}
    net = safe_float(res.get("net"), 0.0)
    buy_hold = safe_float(res.get("buy_hold"), 0.0)
    cell = f"{net:>+8.2f} ({safe_int(res.get('buys'), 0):>2}B/{safe_int(res.get('sells'), 0):>2}S)"
    return net, buy_hold, cell


def print_mt_sweep_header(title_col: str, symbols: list[str], *, col_width: int = 34) -> str:
    width = exact_pos_int(col_width, default=34) or 34
    hdr = f"{title_col:<{width}} | " + " | ".join(f"{sym:>14}" for sym in symbols) + f" | {'PORTFOLIO NET':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))
    return hdr


@contextlib.contextmanager
def preserve_replay_backtest_caches():
    """Save and restore module-level replay backtest caches across a sweep."""
    saved_tape_cache = dict(rrb._TREND_TAPE_CACHE)
    saved_diag = dict(rrb._LAST_RUN_DIAGNOSTICS)
    saved_htp = dict(rrb._LAST_HARD_TP_TOUCHED)
    try:
        yield
    finally:
        rrb._TREND_TAPE_CACHE.clear()
        rrb._TREND_TAPE_CACHE.update(saved_tape_cache)
        rrb._LAST_RUN_DIAGNOSTICS.clear()
        rrb._LAST_RUN_DIAGNOSTICS.update(saved_diag)
        rrb._LAST_HARD_TP_TOUCHED.clear()
        rrb._LAST_HARD_TP_TOUCHED.update(saved_htp)


def print_mt_sweep_row(name: str, cells: list[str], total_net: float, *, col_width: int = 34) -> None:
    width = exact_pos_int(col_width, default=34) or 34
    tot = safe_float(total_net, 0.0)
    print(f"{name:<{width}} | " + " | ".join(f"{c:>14}" for c in cells) + f" | {tot:>+14.2f}")


def print_mt_buy_hold_footer(
    symbols: list[str], bh_by_sym: dict[str, float], hdr: str, *, col_width: int = 34
) -> None:
    vals = [safe_float(bh_by_sym.get(sym) if isinstance(bh_by_sym, dict) else 0.0, 0.0) for sym in symbols]
    bh_cells = [f"{v:>+14.2f}" for v in vals]
    print("-" * len(str(hdr or "")))
    print_mt_sweep_row("Buy & Hold Benchmark", bh_cells, sum(vals), col_width=col_width)


def _normalize_sweep_display_opts(col_width: int, include_buy_hold: bool) -> tuple[int, bool]:
    return exact_pos_int(col_width, default=34) or 34, _parse_bool_flag(include_buy_hold, default=False)


def run_mt_model_sweep_table(
    models,
    symbols: list[str],
    series_map: dict,
    sig_map: dict,
    *,
    title_col: str = "Model Architecture",
    col_width: int = 34,
    include_buy_hold: bool = False,
) -> None:
    """Evaluate a list of ``(model_name, tape_fn)`` definitions across ``symbols`` and print a comparison table."""
    width, inc_bh = _normalize_sweep_display_opts(col_width, include_buy_hold)
    hdr = print_mt_sweep_header(title_col, symbols, col_width=width)

    bh_by_sym: dict[str, float] = {}
    with preserve_replay_backtest_caches():
        for m_name, tape_fn in models:
            cells = []
            tot_net = 0.0
            for sym in symbols:
                net, bh_val, cell = evaluate_mt_model_symbol_cell(
                    sym, series_map[sym], sig_map[sym], tape_fn
                )
                bh_by_sym[sym] = bh_val
                tot_net += net
                cells.append(cell)
            print_mt_sweep_row(m_name, cells, tot_net, col_width=width)

    if inc_bh:
        print_mt_buy_hold_footer(symbols, bh_by_sym, hdr, col_width=width)


def run_streaming_mt_model_sweep(
    models,
    sig_builder,
    *,
    use_365: bool = False,
    title_col: str = "Model Architecture",
    col_width: int = 34,
    include_buy_hold: bool = False,
) -> None:
    """Stream archive series symbol-by-symbol, evaluate ``models``, and print the comparison table."""
    use_dense = _parse_bool_flag(use_365, default=False)
    width, inc_bh = _normalize_sweep_display_opts(col_width, include_buy_hold)
    symbols: list[str] = []
    model_cells: list[list[str]] = [[] for _ in models]
    model_totals: list[float] = [0.0 for _ in models]
    bh_by_sym: dict[str, float] = {}

    with preserve_replay_backtest_caches():
        for sym in list(rrb.SYMBOLS._SECTIONS.keys()):
            series = load_archive_series(sym, use_365=use_dense)
            if not series:
                continue
            symbols.append(sym)
            sig = sig_builder(sym, series)
            for m_idx, (_m_name, tape_fn) in enumerate(models):
                net, bh_val, cell = evaluate_mt_model_symbol_cell(sym, series, sig, tape_fn)
                bh_by_sym[sym] = bh_val
                model_totals[m_idx] += net
                model_cells[m_idx].append(cell)
            rrb._TREND_TAPE_CACHE.clear()
            del sig, series
            gc.collect()

    hdr = print_mt_sweep_header(title_col, symbols, col_width=width)
    for m_idx, (m_name, _fn) in enumerate(models):
        print_mt_sweep_row(m_name, model_cells[m_idx], model_totals[m_idx], col_width=width)
    if inc_bh:
        print_mt_buy_hold_footer(symbols, bh_by_sym, hdr, col_width=width)


def evaluate_dataset(use_365: bool = False):
    label = "DENSE 1-SECOND ARCHIVE (cache_365_price_*.jsonl)" if use_365 else "FULL 329-DAY SPARSE ARCHIVE (cache_sparse_price_*.jsonl)"
    print(f"\n{'=' * 105}")
    print(f"  PORTFOLIO REPLAY BACKTEST: {label}")
    print(f"{'=' * 105}")

    models = [
        ("1. LEGACY (sf+gr)/2 (no NG)",       lambda s: make_tape(s["legacy"], s["eps"], k_eps=0.0)),
        ("2. LEGACY + NG k=0.25",             lambda s: make_tape(s["legacy"], s["eps"], k_eps=0.25)),
        ("3. SW p2 + NG k=0.25",              lambda s: make_tape(s["sw2"], s["eps"], k_eps=0.25)),
        ("4. SW p3 + NG k=0.25",              lambda s: make_tape(s["sw3"], s["eps"], k_eps=0.25)),
        ("5. SW p2 + ScaleInv NG k=0.25",     lambda s: make_tape(s["sw2"], s["eps"], k_eps=0.25, scale_inv=s["scale_inv"])),
        ("6. SW p3 + ScaleInv NG k=0.25",     lambda s: make_tape(s["sw3"], s["eps"], k_eps=0.25, scale_inv=s["scale_inv"])),
        ("7. DIFFERENTIATED (BTC p3/ALT p2)", lambda s: make_tape(s["diff"], s["eps"], k_eps=0.25, scale_inv=s["scale_inv"])),
    ]

    def _build_sig(sym: str, series):
        t0 = time.time()
        sig = compute_all_signals(series, symbol=sym)
        dt = time.time() - t0
        med_m = safe_int(np.median(sig["m"]).item(), 0)
        print(f"  [loaded & vectorized] {sym:<8} (p={pw.step_wols_power_for_symbol(sym):.1f}): "
              f"{len(series):>9,} ticks in {dt:.2f}s (median m={med_m} steps)")
        return sig

    run_streaming_mt_model_sweep(
        models,
        _build_sig,
        use_365=use_365,
        title_col="Model Architecture",
        col_width=34,
        include_buy_hold=True,
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    evaluate_dataset(use_365=False)
    if "--dense" in sys.argv:
        evaluate_dataset(use_365=True)
