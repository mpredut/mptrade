# Kraken strategy robustness research — 2026-08-19

## Decision

Keep the current live configuration unchanged.

The best next shadow candidate is `takeprofit_pct=4.0`. The secondary shadow
candidate is `dca_drop_pct=1.5`. They must be observed separately; the tested
two-factor combination was less stable and is not recommended.

This is exploratory evidence, not a profit guarantee. The windows and
timeframes overlap, so the 69 evaluations below are useful sensitivity checks,
not 69 statistically independent samples.

## What was added

- `offline/runners/fetch_hyperliquid_candles.py` freezes public Hyperliquid spot
  candles into the canonical CSV format accepted by the Kraken replay runner.
- The spot API market is resolved dynamically from `spotMeta`; `@107` is not
  hardcoded.
- The current/incomplete candle is removed, timestamps are sorted and
  deduplicated, and datasets with missing bars are rejected.
- A manifest records venue, market, API market, hashes, range and limitations.
- Generated datasets and reports remain ignored under `offline/results/`.

Implementation commit: `73e097b Add Hyperliquid proxy dataset freezer`.

## Data and evaluation schemes

Fee is `0.26%` per leg unless noted otherwise. Every segment starts with clean
strategy state and uses the same faithful Kraken replay engine as live.

| Source / scheme | TEST windows | Live mean % | Worst % | Worst DD % | Positive |
|---|---:|---:|---:|---:|---:|
| Kraken original 3-fold, 60m/240m/1d | 9 | +0.33 | -7.55 | 8.16 | 5/9 |
| Kraken intraday sensitivity, 5-fold | 10 | +0.20 | -2.02 | 4.58 | 7/10 |
| Kraken daily sensitivity, 5-fold | 5 | +2.32 | -3.31 | 5.35 | 3/5 |
| Hyperliquid common 200 days, 60m/240m/1d | 9 | +5.03 | -0.65 | 7.84 | 7/9 |
| Hyperliquid full history, 240m/1d, auto 3-fold | 6 | +1.76 | -13.99 | 18.25 | 4/6 |
| Hyperliquid full history, 240m, 15 monthly tests | 15 | +0.88 | -6.93 | 12.84 | 10/15 |
| Hyperliquid full history, 1d, 15 monthly tests | 15 | +1.73 | -8.32 | 9.70 | 12/15 |

Kraken contributes 24 window evaluations. Hyperliquid HYPE/USDC spot contributes
45 cross-venue proxy evaluations. The full proxy history contains 3,771 4-hour
bars and 628 daily bars, from 2024-11-29 through 2026-08-18. There were no
timestamp gaps in the frozen datasets.

Hyperliquid is deliberately treated only as a price-path robustness proxy. It
does not reproduce Kraken spread, liquidity, queue position, latency or fills.
The public API exposes at most the most recent 5,000 candles for a market. See
the [official candle API documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint).

## Candidate evidence across all schemes

`Return W/T/L` compares every TEST window with live. `Scheme +/-/=` compares
the mean paired delta inside each of the seven schemes. `DD higher/same/lower`
compares each candidate drawdown with live on the same window.

| Candidate | Mean delta % | Return W/T/L | Scheme +/=/− | DD higher/same/lower | Worst window delta % | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| TP threshold 4% | +0.13 | 7/54/8 | 6/1/0 | 0/56/13 | -1.50 | Best safety candidate; never increased DD |
| DCA drop 1.5% | +0.08 | 11/54/4 | 6/0/1 | 2/54/13 | -0.77 | Small, stable effect; secondary candidate |
| TP threshold 6% | +0.08 | 8/56/5 | 5/2/0 | 9/58/2 | -1.80 | Less attractive risk profile than TP 4% |
| Re-entry 1.5% | +1.03 | 12/50/7 | 4/2/1 | 15/53/1 | -3.27 | Gain concentrated in proxy data; DD often higher |
| Stop-loss off | +1.09 | 8/59/2 | 4/3/0 | 7/59/3 | -1.98 | Reject: maximum DD rose from 18.25% to 41.01% |
| Trailing 2% | +0.15 | 19/37/13 | 5/1/1 | 12/41/16 | -10.30 | Regime-dependent and materially unstable |
| Lower sizing | -0.37 | 19/2/48 | 0/0/7 | 0/2/67 | -3.80 | Valid risk-budget choice, not a better edge |

The current live setup is not dominated at the individual-window level. TP 4%
has the best aggregate risk result, but it still loses return in 8 windows. DCA
1.5% loses in 4 windows. Neither result is strong enough for automatic live
activation.

The pre-registered combination `DCA 1.5% + trailing 2%` should not advance:
it recorded 23 wins, 33 ties and 13 losses, including a worst paired delta of
`-10.59%`. Two individually plausible changes did not combine cleanly.

## Fee stress

The live configuration retained positive mean return in every scheme at the
highest tested fee, `0.50%` per leg:

- Kraken scheme means ranged from `+0.12%` to `+2.16%`.
- Hyperliquid proxy scheme means ranged from `+0.50%` to `+4.56%`.
- Worst-case returns and drawdowns still deteriorated as fees rose; a positive
  average does not remove tail risk.

## Important replay limitations

- Four-hour and daily bars are sensitivity tests, not exact execution replays.
  Live stop and DCA logic observes prices more frequently, while a coarse replay
  may defer decisions until bar close and exaggerate gap/slippage effects.
- OHLC bars do not preserve the intrabar event order. A bar can touch both an
  entry and an exit without revealing which happened first.
- Cross-venue candles validate price-path behavior only.
- Resetting state at every TEST window can differ from a live position carried
  across a boundary.
- Candidate discovery and evaluation reuse related periods. A clean future
  shadow period is still required.

## Recommended next experiment

Run three decision streams on the same live Kraken observations, without placing
additional orders:

1. current live configuration;
2. only `takeprofit_pct=4.0` changed;
3. only `dca_drop_pct=1.5` changed.

Persist timestamped decisions, intended limit prices, fills simulated from the
same Kraken feed, fees, exposure, mark-to-market PnL and drawdown. Evaluate after
at least 30 days and at least 20 behavior-changing events; calendar time alone
is insufficient when most windows tie.

Promotion criteria should be pre-registered before collecting the shadow data:

- positive paired net-PnL delta after fees;
- no worse maximum drawdown;
- no worse loss in the weakest market regime;
- improvement is not produced by one isolated trade;
- zero live-order side effects from the shadow streams.

## Reproduction on dev

```bash
cd /home/predut/binance

python3 offline/runners/fetch_hyperliquid_candles.py \
  --lookback-days 200 \
  --output-dir offline/results/hyperliquid_proxy_20260819/common_200d/datasets

python3 offline/runners/fetch_hyperliquid_candles.py \
  --intervals 240,1440 \
  --output-dir offline/results/hyperliquid_proxy_20260819/full_history/datasets
```

The complete dev artifact bundle is
`kraken-hyperliquid-proxy-backtests-20260819.tar.gz`, SHA-256
`251e67e3cbd966bd7d89e223d1ddd74a48ccb9e6b4abc08598b20b4d25120916`.
