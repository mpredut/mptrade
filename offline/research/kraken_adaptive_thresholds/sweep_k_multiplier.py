#!/usr/bin/env python3
"""
sweep_k_multiplier.py — before promoting the adaptive re-entry (or DCA)
threshold to a real decision, it tests whether the current multiplier K (K_REENTRY=2.0,
K_DCA=1.0, hardcoded in shadow_signals.py) really is the best, or whether a
a larger or smaller K would give a better result — a user request after the initial
results (the adaptive re-entry wins with K=2.0, the adaptive DCA loses with K=1.0).

It reuses EXACTLY the simulation engines from verify_adaptive_dca.py and
verify_adaptive_reentry.py (nothing reimplemented) — it only varies K before
multiplying by vol_1h.

Rulare:  python3 offline/research/kraken_adaptive_thresholds/sweep_k_multiplier.py
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.path.join(ROOT, "kraken") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "kraken"))
if os.path.join(ROOT, "offline", "research", "kraken_adaptive_thresholds") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "offline", "research", "kraken_adaptive_thresholds"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from backtest import fetch_candles
from backtest_adaptive import trailing_vol_series, simulate_variant
from verify_adaptive_reentry import simulate_with_reentry_gate, REAL as REAL_REENTRY, REENTRY_TOLERANCE_PCT
from verify_adaptive_dca import REAL as REAL_DCA

PAIR = "HYPEUSD"
INTERVAL = 60

K_REENTRY_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
K_DCA_SWEEP = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]


def _run_k_sweep(section_title, k_label, ohlc, vol_trail, params, fallback_key, k_values, current_k, sim_fn):
    print(f"--- {section_title} (fix live = {params[fallback_key]}%) ---")
    fixed_arr = np.full(len(ohlc), params[fallback_key])
    m_fix = sim_fn(ohlc, params, fixed_arr)
    tot_fix = m_fix["total"] / params["budget"] * 100
    print(f"  FIX               TOTAL {tot_fix:+7.2f}%  realizat ${m_fix['realized']:+8.2f}  "
          f"cicluri={m_fix['cycles']:<3} maxDD=${m_fix['maxdd']:.2f}")
    best_k, best_tot = None, -1e18
    pad = max(1, 13 - len(k_label))
    for k in k_values:
        arr = np.array([k * v if not np.isnan(v) else np.nan for v in vol_trail])
        m = sim_fn(ohlc, params, arr)
        tot = m["total"] / params["budget"] * 100
        marker = f" <- curent ({current_k})" if k == current_k else ""
        valid = arr[~np.isnan(arr)]
        mean_thresh = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        print(f"  {k_label}={k:<{pad}} TOTAL {tot:+7.2f}%  realizat ${m['realized']:+8.2f}  "
              f"cicluri={m['cycles']:<3} maxDD=${m['maxdd']:.2f}  medie_prag={mean_thresh:.2f}%{marker}")
        if tot > best_tot:
            best_k, best_tot = k, tot
    print(f"  => the best {k_label} in the sweep: {best_k} (TOTAL {best_tot:+.2f}%)")
    return tot_fix, best_k, best_tot


def main():
    ohlc = fetch_candles(PAIR, INTERVAL)
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100
    print(f"=== {PAIR} interval={INTERVAL}m ({len(ohlc)} bare = ~{len(ohlc)/24:.1f} zile) — buy&hold {bh:+.1f}% ===\n")
    vol_trail = trailing_vol_series(closes)

    tot_fix_re, best_k_re, best_tot_re = _run_k_sweep(
        "SWEEP REINTRARE", "K_REENTRY", ohlc, vol_trail, REAL_REENTRY,
        "reentry_fallback", K_REENTRY_SWEEP, 2.0, simulate_with_reentry_gate,
    )
    print()
    tot_fix_dca, best_k_dca, best_tot_dca = _run_k_sweep(
        "SWEEP DCA", "K_DCA", ohlc, vol_trail, REAL_DCA,
        "drop_fallback", K_DCA_SWEEP, 1.0, simulate_variant,
    )

    print(f"\n=== REZUMAT ===")
    print(f"Reintrare: FIX {tot_fix_re:+.2f}%  |  cel mai bun adaptiv (K={best_k_re}) {best_tot_re:+.2f}%")
    print(f"DCA:       FIX {tot_fix_dca:+.2f}%  |  cel mai bun adaptiv (K={best_k_dca}) {best_tot_dca:+.2f}%")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
