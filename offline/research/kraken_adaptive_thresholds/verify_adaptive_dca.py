#!/usr/bin/env python3
"""
verify_adaptive_dca.py — merita promovat pragul de DCA adaptiv (shadow, K_DCA *
the realised vol_1h, already computed in shadow_signals.py but used ONLY as a log in
kraken/strategy.py) la decizie REALA, in locul pragului FIX (STRAT_DCA_DROP_PCT)?

It reuses EXACTLY the simulation engine from kraken/backtest_adaptive.py
(simulate_variant/trailing_vol_series/fetch_candles) — it does not rewrite the logic.
It SKIPS the Chronos part (an ML model — it would load memory on the machine
real active ones, useless for this specific question: fixed versus shadow).

Parameters = the REAL values from kraken/config.env (NOT the defaults of the
original script, which are guessed or stale — exactly the mistake caught in a
session: the first run with the wrong parameters showed a large gain for the
adaptive variant, and then it reversed completely with the real values).

Rulare:  python3 offline/research/kraken_adaptive_thresholds/verify_adaptive_dca.py
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.path.join(ROOT, "kraken") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "kraken"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from backtest import fetch_candles
from backtest_adaptive import trailing_vol_series, simulate_variant, report
import shadow_signals as ss

PAIR = "HYPEUSD"
INTERVAL = 60  # minute per bara (1h) — rezolutia formulei vol_1h_pct

# REAL values from kraken/config.env (NOT the script defaults!)
REAL = dict(
    entry=650, dca=325, disc=0.8, tp=5.0, maxdca=10, budget=3900, fee=0.26, sl=7.0,
    drop_fallback=1.0,   # STRAT_DCA_DROP_PCT actual
)


def run():
    ohlc = fetch_candles(PAIR, INTERVAL)
    if not ohlc:
        print(f"=== {PAIR}: no candles returned ===")
        return {}, {}
    print(f"=== {PAIR} interval={INTERVAL}m ({len(ohlc)} bare = ~{len(ohlc)/24:.1f} zile) ===")
    closes = [x[3] for x in ohlc]
    bh = ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] > 0 else 0.0
    print(f"buy&hold pe perioada: {bh:+.1f}%")

    vol_trail = trailing_vol_series(closes)
    n_valid = np.sum(~np.isnan(vol_trail))
    print(f"vol_1h_pct trailing disponibil pt {n_valid}/{len(closes)} bare (primele 24 sunt warm-up)")

    fixed_arr = np.full(len(ohlc), REAL["drop_fallback"])
    shadow_arr = np.array([ss.K_DCA * v if not np.isnan(v) else np.nan for v in vol_trail])
    valid_shadow = shadow_arr[~np.isnan(shadow_arr)]
    mean_s = float(np.mean(valid_shadow)) if len(valid_shadow) else float("nan")
    min_s = float(np.min(valid_shadow)) if len(valid_shadow) else float("nan")
    max_s = float(np.max(valid_shadow)) if len(valid_shadow) else float("nan")

    print(f"\n(K_DCA={ss.K_DCA}, prag FIX real={REAL['drop_fallback']}%, "
          f"medie adaptiv-shadow={mean_s:.2f}%, "
          f"min={min_s:.2f}% max={max_s:.2f}%)")

    m_fixed = simulate_variant(ohlc, REAL, fixed_arr)
    m_shadow = simulate_variant(ohlc, REAL, shadow_arr)

    report("FIX (live azi)", m_fixed, REAL["budget"])
    report("adaptiv-shadow", m_shadow, REAL["budget"])
    return m_fixed, m_shadow


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
