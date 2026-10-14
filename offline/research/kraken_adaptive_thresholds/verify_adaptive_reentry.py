#!/usr/bin/env python3
"""
verify_adaptive_reentry.py — merita promovat pragul de REINTRARE adaptiv
(a shadow value, K_REENTRY * vol_1h — see _shadow_reentry_line in kraken/strategy.py,
today a LOG ONLY) to a REAL decision, in place of the FIXED threshold (STRAT_REENTRY_DROP_PCT)?

Found during the investigation: NEITHER backtest.py NOR backtest_adaptive.py
modelled the re-entry barrier — once a position closes (take-profit or
a stop-loss), the original simulators re-entered IMMEDIATELY on the next bar, without
any wait. The REAL strategy (kraken/strategy.py, the step() function) waits
explicitly for the price to fall below last_sell_price*(1-reentry_pct/100) (with
the are_close tolerance) before re-entering.

23 Jul: the mechanism (written at first ONLY here, so as not to push into the
ceva netestat) a fost MERGE-uit inapoi in backtest.simulate() ca parametru
an OPTIONAL `reentry_arr` (default None = the old behaviour, unchanged) —
the reason is to stop having TWO copies of the DCA/TP/SL engine that can drift
apart over time (see offline/research/BACKTEST_CANDIDATES.md and the session discussion
about unifying the backtests). This script stays a thin wrapper over the
official tool, with the REAL values from kraken/config.env — it no longer
redefineste motorul de simulare.

Rulare:  python3 offline/research/kraken_adaptive_thresholds/verify_adaptive_reentry.py
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.path.join(ROOT, "kraken") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "kraken"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from backtest import fetch_candles, simulate as simulate_with_reentry_gate
from backtest_adaptive import trailing_vol_series, report
import shadow_signals as ss

PAIR = "HYPEUSD"
INTERVAL = 60

REAL = dict(
    entry=650, dca=325, disc=0.8, tp=5.0, maxdca=10, budget=3900, fee=0.26, sl=7.0,
    drop=1.0,                     # STRAT_DCA_DROP_PCT — fixed, unchanged (not the subject of this test).
    reentry_fallback=2.2,         # STRAT_REENTRY_DROP_PCT actual
    reentry_tolerance_pct=0.05,   # the current STRAT_REENTRY_TOLERANCE_PCT (now passed through P, not global)
)
REENTRY_TOLERANCE_PCT = REAL["reentry_tolerance_pct"]  # The name is kept; it is used only in the print below.


def run():
    ohlc = fetch_candles(PAIR, INTERVAL)
    print(f"=== {PAIR} interval={INTERVAL}m ({len(ohlc)} bare = ~{len(ohlc)/24:.1f} zile) ===")
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100
    print(f"buy&hold pe perioada: {bh:+.1f}%")

    vol_trail = trailing_vol_series(closes)
    fixed_arr = np.full(len(ohlc), REAL["reentry_fallback"])
    shadow_arr = np.array([ss.K_REENTRY * v if not np.isnan(v) else np.nan for v in vol_trail])

    print(f"\n(K_REENTRY={ss.K_REENTRY}, prag FIX real={REAL['reentry_fallback']}%, "
          f"toleranta={REENTRY_TOLERANCE_PCT}%, medie adaptiv-shadow={np.nanmean(shadow_arr):.2f}%, "
          f"min={np.nanmin(shadow_arr):.2f}% max={np.nanmax(shadow_arr):.2f}%)")

    m_fixed = simulate_with_reentry_gate(ohlc, REAL, fixed_arr)
    m_shadow = simulate_with_reentry_gate(ohlc, REAL, shadow_arr)

    for name, m in (("FIX (live azi)", m_fixed), ("adaptiv-shadow", m_shadow)):
        tot = m["total"]/REAL["budget"]*100
        wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
        print(f"  [{name:16s}] TOTAL {tot:+7.2f}%  realizat ${m['realized']:+8.2f}  "
              f"cicluri={m['cycles']:<3} win-rate={wr:3.0f}%  maxDD=${m['maxdd']:.2f}  "
              f"bare-blocate={m['blocked_ticks']:<4} pozitie deschisa={m['open_qty']:.4f}")
    return m_fixed, m_shadow


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
