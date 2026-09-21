# RSI / Bollinger Bands review (2026-09-14)

**Question (from the owner):** would RSI and Bollinger Bands be useful anywhere in the
strategies?

**Verdict: no.** Tested in both plausible roles — as a trigger and as a DCA filter —
and both were rejected on 329 days of real history with a first-half/second-half overfit
guard. Do not add them. The existing signal stack (Kalman trend/velocity, volatility-
adaptive DCA/reentry thresholds, the `dca_trend_brake` regime gate, SMA100) already
covers what they would nominally provide, and the data shows it does so better.

## Context: what already exists

- **Momentum/trend:** `shadow_signals.KalmanTrend` (velocity + Schmitt hysteresis), used
  by tradeall's TrendState.
- **Volatility -> adaptive thresholds:** `shadow_signals.vol_1h_pct` feeds
  `adaptive_thresholds`, scaling DCA/reentry drop % by realized volatility (this is
  Bollinger-adjacent already).
- **Long-term trend:** SMA100 (trailing rebuy `auto`), `detect_long_term_trend`.
- **Regime:** `market_regime.py`, `dca_trend_brake` / `_regime_matches` skip DCA in a
  confirmed bear.

Neither RSI nor Bollinger existed in the code. RSI (a bounded oscillator) was the more
orthogonal candidate; Bollinger %B mostly duplicates the volatility-adaptive thresholds.

## Experiment 8 — RSI / %B as a TRIGGER (rejected)

`offline/research/tradeall_trigger_gate/experiment_rsi_bollinger.py`. A contrarian
signal (buy oversold / lower band, sell overbought / upper band) via `ta._fire_order`,
replayed on the sparse 329-day history. Variants `RSI14_30_70`, `RSI14_20_80`, `PB20_k2`,
`RSI_AND_PB`, on BTC + TAO, full + both halves.

- **No variant beats buy & hold on the full span AND both halves.** Every one shows the
  same shape: deep loss in H1, profit in H2 — e.g. TAO `RSI14_30_70` full +13,498 but
  H1 **-16,753** / H2 +8,459 USDC. The positive full-period numbers are entirely
  H2-driven and would have been wiped out in H1.
- **It is a regime bet, not an edge.** Mean-reversion got destroyed fighting the H1
  downtrend (catching knives) and harvested the H2 chop. The H1/H2 split is what exposes
  it — the full span alone would have produced a false positive.
- **Overtrading.** Even with fire-once + a 30-min retry cooldown the threshold flips
  constantly: 10k-30k trades; on the best run fees were ~29% of gross.

## Experiment 9 — RSI as a DCA FILTER (rejected)

`offline/research/tradeall_trigger_gate/experiment_rsi_dca_filter.py`. Keep spot_dca's
DCA logic; only allow a DCA buy when RSI is oversold (wrap `Strategy.step` to feed a
Wilder RSI and `spot_dca_rules.dca_price_hit` to AND in `RSI < threshold`; baseline
leaves the tracker None => unmodified strategy). 329-day history resampled to hourly
OHLC, HL live profile (entry 350 / dca 100 / drop 2% / max 7 / TP 5% / budget 1050 /
stop 20), fee 0.26 (deliberately favouring the gate, which skips DCAs and their fees).
Baseline vs RSI<30 vs RSI<40 on BTC / TAO / ARB, full + both halves.

Delta = gated total - baseline total (USDC, 329 days):

| symbol | gate | full | H1 | H2 |
|---|---|--:|--:|--:|
| BTC | rsi30 | -54.3 | -20.6 | +7.6 |
| BTC | rsi40 | -113.3 | -76.3 | -20.0 |
| TAO | rsi30 | **-331.6** | -69.4 | -165.9 |
| TAO | rsi40 | -176.2 | +3.9 | -148.8 |
| ARB | rsi30 | +14.9 | (no H1) | +14.9 |

- **On BTC and TAO (the only full-length datasets) the gate consistently loses vs just
  averaging down on every -2% dip.** spot_dca's DCA *is* mean-reversion accumulation —
  buy the dip, let the +5% TP catch the bounce. Requiring RSI-oversold skips dips that
  are not deeply oversold but still bounce to TP, removing more winning DCAs than losing
  ones. The "do not average into a crash" case is already handled better by
  `dca_trend_brake`.
- **ARB is inconclusive:** its sparse history is only 145 hourly bars (all in H2, no
  H1), so its small positive delta is noise and cannot clear the full+both-halves bar.

## Reproduce

Both scripts are isolated (they monkeypatch in memory; they never modify live code or
touch the network) and are meant to run on the DEV/backtest box, not PROD:

```bash
ssh backtest 'git -C /home/predut/binance pull --ff-only && \
  /home/predut/binance/myenv/bin/python3 \
  /home/predut/binance/offline/research/tradeall_trigger_gate/experiment_rsi_bollinger.py'
# and .../experiment_rsi_dca_filter.py
```

This closes the RSI/Bollinger question. It also extends the tradeall trigger-gate
record: no signal change tried (Experiments 1-9) beats buy & hold or the current live
configuration.
