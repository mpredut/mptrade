#!/usr/bin/env python3
"""
backtest_adaptive.py — decision test, not merely a statistical test: compare the
fixed DCA threshold (--drop, currently 2.0% for HYPEUSD), the volatility-adaptive
threshold (K_DCA * historical vol_1h, already implemented in shadow_signals.py but
only logged by kraken/strategy.py), and a threshold based on volatility forecast by
Chronos (zero-shot; see forecast/vol_chronos.py, which beat persistence on MAE and
direction using real data).

The relevant question is not whether the forecast is statistically better than the
baseline (it is, modestly), but whether using it for re-entry/DCA would have changed
real financial results. This reuses the exact simulation engine from backtest.py
(DCA/TP/SL/fees/budget) and changes only the drop-threshold source to isolate its effect.

Chronos runs at a reduced cadence (--refresh hours, default 6h), as it would live in
a separate process rather than kraken_bot, to avoid burdening the memory-constrained
machine that hosts the live bots.

  python3 backtest_adaptive.py --pair HYPEUSD --interval 60
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)
try:
    from kraken.backtest import _bt_params, fetch_candles
    from kraken import replay as rp
except ImportError:
    from backtest import _bt_params, fetch_candles  # noqa: E402
    import replay as rp  # noqa: E402

from offline.backtests.metrics import format_performance_summary
import shadow_signals as ss  # noqa: E402  (reuse the exact K_DCA/vol_1h_pct formula)

WIN = 24            # trailing hours for realized volatility; same window as vol_chronos.py


def _nanmean(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def trailing_vol_series(closes: list[float], sample_rate_sec: float = 3600.0) -> np.ndarray:
    """Compute causal vol_1h_pct per bar using the exact shadow-signals formula."""
    if closes is None or isinstance(closes, (bool, int, float, str)):
        return np.empty(0, dtype=float)
    if sample_rate_sec is None or isinstance(sample_rate_sec, bool):
        sr = 3600.0
    else:
        try:
            sr_f = float(sample_rate_sec)
            sr = sr_f if (np.isfinite(sr_f) and sr_f > 0) else 3600.0
        except (TypeError, ValueError):
            sr = 3600.0
    out = np.full(len(closes), np.nan)
    for i in range(WIN, len(closes)):
        window = closes[i - WIN:i + 1]
        v = ss.vol_1h_pct(window, sample_rate_sec=sr)
        out[i] = v if v is not None else np.nan
    return out


def chronos_forecast_series(vol_trail: np.ndarray, refresh: int, horizon: int) -> np.ndarray:
    """Forecast trailing volatility with Chronos at the reduced refresh cadence.

    Between refreshes, retain the last forecast just as a bot would retain a
    signal read from a periodically updated file produced by a separate daemon.
    """
    from chronos import ChronosPipeline
    import torch
    try:
        ref_f = float(refresh) if refresh is not None and not isinstance(refresh, bool) else 6.0
        refresh_eff = max(1, int(ref_f)) if np.isfinite(ref_f) else 6
    except (TypeError, ValueError):
        refresh_eff = 6
    try:
        hor_f = float(horizon) if horizon is not None and not isinstance(horizon, bool) else 24.0
        horizon_eff = max(1, int(hor_f)) if np.isfinite(hor_f) else 24
    except (TypeError, ValueError):
        horizon_eff = 24
    print("  incarc amazon/chronos-t5-tiny...")
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu",
                                            torch_dtype=torch.float32)
    out = np.full(len(vol_trail), np.nan)
    last = None
    for i in range(len(vol_trail)):
        if np.isnan(vol_trail[i]):
            continue
        if last is None or i % refresh_eff == 0:
            ctx = vol_trail[max(0, i - 511):i + 1]
            ctx = ctx[~np.isnan(ctx)]
            if len(ctx) < 8:
                continue
            fc = pipe.predict([torch.tensor(ctx, dtype=torch.float32)], prediction_length=horizon_eff)
            last = float(np.median(fc[0].numpy()))
        out[i] = last
    return out


def simulate_variant(ohlc, P, drop_arr):
    """Match ``backtest.simulate`` but dynamically apply ``drop_arr`` per bar.

    Delegates to the faithful replay engine (rp.run_replay) which runs
    strategies.spot_engine.Strategy directly with dynamic per-bar DCA drop thresholds,
    eliminating parity bugs and duplicate simulation drift.
    """
    drop_fb = P.get("drop_fallback")
    if drop_fb is None:
        drop_fb = P.get("drop")
    if drop_fb is None:
        drop_fb = 2.0
    re_drop = P.get("reentry_drop")
    if re_drop is None:
        re_drop = P.get("reentry")
    if re_drop is None:
        re_drop = 0.0
    params = _bt_params(P, {
        "dca_drop_pct": drop_fb,
        "reentry_drop_pct": re_drop,
    })
    return rp.run_replay(
        ohlc,
        params,
        fee_pct=P.get("fee") if P.get("fee") is not None else 0.25,
        bar_minutes=P.get("interval") if P.get("interval") is not None else 60,
        dca_drop_series=drop_arr,
    )


def report(name, m, budget):
    try:
        bud_f = float(budget) if budget is not None and not isinstance(budget, bool) else 0.0
    except (TypeError, ValueError):
        bud_f = 0.0
    tot = m["total"] / bud_f * 100 if (np.isfinite(bud_f) and bud_f > 0) else 0.0
    wr = 100 * m["wins"] / m["cycles"] if m.get("cycles") else 0
    print(f"  [{name:16s}] TOTAL {tot:+7.2f}%  realizat ${m['realized']:+8.2f}  "
          f"cicluri={m['cycles']:<3} win-rate={wr:3.0f}%  maxDD=${m['maxdd']:.2f}  "
          f"pozitie deschisa={m['open_qty']:.4f}")
    for line in format_performance_summary(m, indent="    "):
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix vs adaptiv-shadow vs adaptiv-Chronos, pe DCA real.")
    ap.add_argument("--pair", default="HYPEUSD")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--entry", type=float, default=100); ap.add_argument("--dca", type=float, default=50)
    ap.add_argument("--disc", type=float, default=0.2); ap.add_argument("--drop", type=float, default=2.0,
                    help="pragul FIX actual, live (baseline de batut)")
    ap.add_argument("--tp", type=float, default=1.9); ap.add_argument("--maxdca", type=int, default=10)
    ap.add_argument("--budget", type=float, default=1000); ap.add_argument("--fee", type=float, default=0.25)
    ap.add_argument("--sl", type=float, default=10.0)
    ap.add_argument("--refresh", type=int, default=6, help="hours between Chronos updates (the daemon's cadence)")
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()

    if args.budget <= 0 or args.interval <= 0:
        print("! budget and interval must be > 0"); return 1
    ohlc = fetch_candles(args.pair, args.interval)
    if len(ohlc) < 40:
        print(f"! not enough data ({len(ohlc)})"); return 1
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0.0
    print(f"=== {args.pair} interval={args.interval}m ({len(ohlc)} bare) — buy&hold {bh:+.1f}% ===")

    vol_trail = trailing_vol_series(closes, sample_rate_sec=args.interval * 60.0)
    print(f"  vol_1h_pct trailing: available from bar {WIN} (of {len(closes)})")
    chronos_vol = chronos_forecast_series(vol_trail, args.refresh, args.horizon)

    base = dict(entry=args.entry, dca=args.dca, disc=args.disc, tp=args.tp,
                maxdca=args.maxdca, budget=args.budget, fee=args.fee, sl=args.sl,
                drop_fallback=args.drop, interval=args.interval)

    fixed_arr = np.full(len(ohlc), args.drop)
    shadow_arr = np.array([ss.K_DCA * v if not np.isnan(v) else np.nan for v in vol_trail])
    chronos_arr = np.array([ss.K_DCA * v if not np.isnan(v) else np.nan for v in chronos_vol])

    m_fixed = simulate_variant(ohlc, base, fixed_arr)
    m_shadow = simulate_variant(ohlc, base, shadow_arr)
    m_chronos = simulate_variant(ohlc, base, chronos_arr)

    print(f"  (K_DCA={ss.K_DCA}, prag fix={args.drop}%, medie adaptiv-shadow="
          f"{_nanmean(shadow_arr):.2f}%, medie adaptiv-chronos={_nanmean(chronos_arr):.2f}%)")
    report("FIX (live azi)", m_fixed, args.budget)
    report("adaptiv-shadow", m_shadow, args.budget)
    report("adaptiv-chronos", m_chronos, args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
