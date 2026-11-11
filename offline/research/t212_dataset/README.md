# Trading212 -- frozen datasets and the live baseline

Date: 2026-08-20

The CSVs are canonical Yahoo copies, without the trailing tail of candles still forming. The
hashes and the ranges are in `manifest.json`. Each profile's configuration is read directly
from `212trading/config.<profile>.env`; the runner uses the same `Strategy.step()` as live and
never contacts the T212 order API.

## Results

| Profile | Data | Central mean/worst/DD | Stress mean/worst/DD | Central cycles | Verdict |
|---|---|---:|---:|---:|---|
| NVDA | 501 bars 1d, 2 years | +0.536% / +0.450% / 0.914% | +0.458% / +0.360% / 1.037% | 9 | a positive baseline, but only 3 folds |
| RGNT | 176 bars 1d, since listing | -4.771% / -13.778% / 18.579% | -4.771% / -13.778% / 18.579% | 1 | the configuration needs re-evaluation; extremely thin evidence |
| SPCX | 3276 bars 5m, 59 days | +1.438% / -0.621% / 1.955% | +1.440% / -0.621% / 1.952% | 0 | characterisation: 3 folds, 0 cycles, a negative worst fold |

Central: 10bps spread, 15bps MARKET slippage, at most a 75% LIMIT fill per bar, worst-case
intrabar. Stress: 20bps, 30bps, at most 50%, worst-case.

Every profile declares `STRAT_CURRENCY=USD`: the budgets and quantities are expressed in USD.
The T212 endpoint `/equity/account/info` confirmed read-only that `currencyCode=RON`, so
`STRAT_FX_FEE_PCT=0.15` remains correct and is included in the baseline. A historical FX
series is not needed to size a budget fixed in USD; it would become necessary if the
strategy's budgets were expressed in RON or EUR.

The runner now includes an `evidence_gate` that explicitly separates sample problems
(`folds`, days of history, closed cycles) from risk signals (`negative_worst_fold`). SPCX
remains `characterization_only_with_risk_flags`: the 5m window was extended from 31 to 59
days, the practical maximum Yahoo offers, but the asset was listed recently and the strategy
has not closed a single cycle yet. The positive mean is mostly mark-to-market on open
inventory, not realised profit; it does not justify changing the parameters.

The provider's limit no longer truncates the history permanently: on refresh, `--seed-dataset`
merges the frozen CSV with the last 59 days from Yahoo and produces a new CSV without
overwriting the old one. The SPCX window will therefore grow over time:

```bash
.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile spcx --range 59d --interval 5m \
  --seed-dataset offline/research/t212_dataset/spcx/datasets/SPCX_5m_1cfe20146366.csv
```

## Reproduction

```bash
.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile nvda --interval 1d \
  --dataset offline/research/t212_dataset/nvda/datasets/NVDA_1d_9e8cbd3c6ff5.csv \
  --spread-bps 10 --market-slippage-bps 15 \
  --partial-fill-ratio 0.75 --intrabar-policy worst_case

.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile rgnt --interval 1d \
  --dataset offline/research/t212_dataset/rgnt/datasets/RGNT_1d_b67a2932ddd6.csv \
  --spread-bps 10 --market-slippage-bps 15 \
  --partial-fill-ratio 0.75 --intrabar-policy worst_case

.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile spcx --interval 5m \
  --dataset offline/research/t212_dataset/spcx/datasets/SPCX_5m_1cfe20146366.csv \
  --spread-bps 10 --market-slippage-bps 15 \
  --partial-fill-ratio 0.75 --intrabar-policy worst_case
```

For the stress scenario, switch to `--spread-bps 20 --market-slippage-bps 30
--partial-fill-ratio 0.50`; the dataset and the other options stay identical.
