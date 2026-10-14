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


def main():
    ohlc = fetch_candles(PAIR, INTERVAL)
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100
    print(f"=== {PAIR} interval={INTERVAL}m ({len(ohlc)} bare = ~{len(ohlc)/24:.1f} zile) — buy&hold {bh:+.1f}% ===\n")
    vol_trail = trailing_vol_series(closes)

    print(f"--- SWEEP REINTRARE (fix live = {REAL_REENTRY['reentry_fallback']}%) ---")
    fixed_arr_re = np.full(len(ohlc), REAL_REENTRY["reentry_fallback"])
    m_fix_re = simulate_with_reentry_gate(ohlc, REAL_REENTRY, fixed_arr_re)
    tot_fix_re = m_fix_re["total"] / REAL_REENTRY["budget"] * 100
    print(f"  FIX               TOTAL {tot_fix_re:+7.2f}%  realizat ${m_fix_re['realized']:+8.2f}  "
          f"cicluri={m_fix_re['cycles']:<3} maxDD=${m_fix_re['maxdd']:.2f}")
    best_k_re, best_tot_re = None, -1e18
    for k in K_REENTRY_SWEEP:
        arr = np.array([k * v if not np.isnan(v) else np.nan for v in vol_trail])
        m = simulate_with_reentry_gate(ohlc, REAL_REENTRY, arr)
        tot = m["total"] / REAL_REENTRY["budget"] * 100
        marker = " <- curent (2.0)" if k == 2.0 else ""
        print(f"  K_REENTRY={k:<4} TOTAL {tot:+7.2f}%  realizat ${m['realized']:+8.2f}  "
              f"cicluri={m['cycles']:<3} maxDD=${m['maxdd']:.2f}  medie_prag={np.nanmean(arr):.2f}%{marker}")
        if tot > best_tot_re:
            best_k_re, best_tot_re = k, tot
    print(f"  => the best K_REENTRY in the sweep: {best_k_re} (TOTAL {best_tot_re:+.2f}%)")

    print(f"\n--- SWEEP DCA (fix live = {REAL_DCA['drop_fallback']}%) ---")
    fixed_arr_dca = np.full(len(ohlc), REAL_DCA["drop_fallback"])
    m_fix_dca = simulate_variant(ohlc, REAL_DCA, fixed_arr_dca)
    tot_fix_dca = m_fix_dca["total"] / REAL_DCA["budget"] * 100
    print(f"  FIX               TOTAL {tot_fix_dca:+7.2f}%  realizat ${m_fix_dca['realized']:+8.2f}  "
          f"cicluri={m_fix_dca['cycles']:<3} maxDD=${m_fix_dca['maxdd']:.2f}")
    best_k_dca, best_tot_dca = None, -1e18
    for k in K_DCA_SWEEP:
        arr = np.array([k * v if not np.isnan(v) else np.nan for v in vol_trail])
        m = simulate_variant(ohlc, REAL_DCA, arr)
        tot = m["total"] / REAL_DCA["budget"] * 100
        marker = " <- curent (1.0)" if k == 1.0 else ""
        print(f"  K_DCA={k:<7} TOTAL {tot:+7.2f}%  realizat ${m['realized']:+8.2f}  "
              f"cicluri={m['cycles']:<3} maxDD=${m['maxdd']:.2f}  medie_prag={np.nanmean(arr):.2f}%{marker}")
        if tot > best_tot_dca:
            best_k_dca, best_tot_dca = k, tot
    print(f"  => the best K_DCA in the sweep: {best_k_dca} (TOTAL {best_tot_dca:+.2f}%)")

    print(f"\n=== REZUMAT ===")
    print(f"Reintrare: FIX {tot_fix_re:+.2f}%  |  cel mai bun adaptiv (K={best_k_re}) {best_tot_re:+.2f}%")
    print(f"DCA:       FIX {tot_fix_dca:+.2f}%  |  cel mai bun adaptiv (K={best_k_dca}) {best_tot_dca:+.2f}%")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
