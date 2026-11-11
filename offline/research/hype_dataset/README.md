# The frozen HYPE dataset (a Hyperliquid proxy)

A reproducible dataset for re-running any Kraken/HYPE strategy candidate INSTANTLY, without
depending on a live fetch (Kraken's public OHLC only gives ~120 days; here we have ~628).

- `HYPEUSDC_240m_hlspot.csv` -- 3772 bars of 4h (~628 days)
- `HYPEUSDC_1440m_hlspot.csv` -- 628 bars of 1 day
- `manifest.json` -- source, sha256 hash, fetch date

**Source:** Hyperliquid public OHLC (`api.hyperliquid.xyz`), HYPE/USDC spot, through
`offline/runners/fetch_hyperliquid_candles.py`. A **cross-venue PROXY** for HYPE's price
movement -- NOT Kraken execution (the real fills differ). Good for robustness, not for
absolute numbers.

## Reproducible candidate verification

The declared scheme for the HYPE comparison is fixed: 720 TRAIN bars, 180 VALIDATION, 90
TEST, a step of 90, a 40-bar signal warm-up and a fee of 0.26% per leg. For the 3,772 bars of
4h this yields exactly 31 TEST windows. The warm-up loads only the SMA, volatility and trend;
the position, the orders and the P&L start clean in every segment.

```bash
.venv/bin/python offline/runners/kraken_walk_forward_baseline.py \
  --intervals 240 \
  --dataset 240=offline/research/hype_dataset/HYPEUSDC_240m_hlspot.csv \
  --train 720 --validation 180 --test 90 --step 90 --warmup 40 \
  --output-dir offline/results/hype_long_31

.venv/bin/python offline/runners/kraken_walk_forward_compare.py \
  offline/results/hype_long_31/baseline_HYPEUSD_<timestamp>.json \
  --candidate-set hype-240
```

The contract test checks the files against the manifest, the baseline records the hash, and
the comparator re-verifies it before evaluating the candidates. The figures below were
regenerated after the fill/warm-up fixes, on the scheme declared above. The regenerated
report remains the source of truth if the engine changes.

## The versioned financial baseline

`financial_baseline_v1.json` is the quantitative baseline of the current live configuration.
It uses the same 31 TEST windows, but separates maker and taker fees and runs the `central`
and `stress` scenarios with spread, slippage, partial fills and worst-case intrabar ordering.
The human-readable report is in `chatgpt_agent_work/HYPE_FINANCIAL_BENCHMARK.md`.

This does not replace the golden: the golden detects changes in decisions, while the financial
baseline quantifies the effect on return, USD, drawdown, CVaR, exposure and the
bull/bear/sideways regimes. The artefact is verified with:

```bash
.venv/bin/python offline/runners/kraken_financial_benchmark.py \
  --verify offline/research/hype_dataset/financial_baseline_v1.json \
  --output /tmp/hype_financial_verify.json \
  --markdown /tmp/hype_financial_verify.md
```

## Reproducible result (31 OOS windows, 40-bar warm-up, 0.26% fee per leg)

BASE v2: mean **+0.777%**, worst window `-9.123%`, max DD `12.305%`.

| Candidate | delta  mean vs base | W/T/L | worst fold | max DD | verdict |
|---|---|---|---|---|---|
| **original overlay** (topup 2000/trail 5) | +0.530pp | 15/2/14 | **-12.612%** | **19.212%** | REJECTED -- the mean comes with a worse tail and drawdown |
| **overlay650t8** | +0.637pp | 15/0/16 | **-7.373%** | **9.105%** | a shadow candidate; selected on the proxy, and it still loses 16 of 31 pairs |
| **A** adaptive trailing | -0.368pp | 11/12/8 | -9.460% | 12.305% | no promotion: a weaker mean, with no tail advantage |
| **B** downtrend DCA brake | -0.457pp | 5/14/12 | **-7.829%** | 10.113% | it reduces tail and drawdown, but sacrifices return |
| **tp4** (TP 5->4) | +0.054pp | 3/25/3 | -7.398% | 10.113% | marginal -- almost always identical |
| **dca15** (DCA 1.25->1.5) | +0.005pp | 7/21/3 | -8.858% | 12.213% | effectively inert |

These windows overlap through train/validation and come from the same price path; they are
not 31 statistically independent experiments. The solid conclusion is a screening one: A does
not justify promotion, the original overlay amplifies risk, and B trades return for
protection. `overlay650t8` needs Kraken/shadow confirmation because it was chosen after
exploring the same proxy.

## Sensitivity to conservative execution

An exploratory, uncalibrated scenario (`20bps spread`, `30bps` market slippage, at most 50%
of a limit order per bar, worst-case intrabar) changes the base:

- mean: `+0.777%` -> `+0.260%`;
- worst fold: `-9.123%` -> `-9.485%`;
- fill events: `212` -> `327` (partial fills are separate events).

Only one TEST bar out of the 31 windows had an eligible BUY and SELL at the same time; the
degradation comes predominantly from the spread and from execution in tranches. Under this
stress, `overlay650t8` stays above base on the mean (`+0.338%`) but loses 17 of 31
comparisons; it gains no additional reason for promotion.

## Conclusion (aligned with the Codex revalidation of 19 Aug)
- **The original overlay: rejected** -- the exact reason is **instability plus tail risk**, not
  a uniform loss on the mean (it wins 15 of 31 comparisons, but its tail is far worse than base).
- **A / tp4 / dca15:** no sufficient advantage, so shadow only, no promotion.
- **B:** OFF on return, but its **tail/drawdown protection** angle is real and under-explored.
- **Live unchanged.** The promotion threshold: at least 30 days plus 20 divergence events in shadow.
