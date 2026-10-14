# Offline runners

Orchestration of the backtests and of the prod -> dev flow. The scripts are run from the
repository root, or through the absolute paths kept up to date in `systemd/crontab.prod.txt`.

For the full architectural specification, multi-venue execution matrix, and replay mechanics, see
[`docs/UNIFIED_BACKTEST_AND_OFFLINE_ARCHITECTURE.md`](../../docs/UNIFIED_BACKTEST_AND_OFFLINE_ARCHITECTURE.md).

## Unified Walk-Forward Facade (`walk_forward.py`)

All venues and modes are accessible through a single linear dispatcher:

```bash
# Baseline evaluation
.venv/bin/python offline/runners/walk_forward.py --venue binance --symbol BTCUSDT --intervals 240
.venv/bin/python offline/runners/walk_forward.py --venue kraken --intervals 240 --dataset 240=path/to/candles.csv
.venv/bin/python offline/runners/walk_forward.py --venue hyperliquid --coin HYPE --direction long --intervals 240
.venv/bin/python offline/runners/walk_forward.py --venue t212 --profile nvda --range 2y --interval 1d

# Candidate robustness comparison
.venv/bin/python offline/runners/walk_forward.py --mode compare offline/results/binance_wf/BTCUSDT_240m_baseline.json
.venv/bin/python offline/runners/walk_forward.py --mode compare offline/results/kraken_wf/baseline_HYPEUSD_*.json --candidate-set hype-240
.venv/bin/python offline/runners/walk_forward.py --mode compare offline/results/hyperliquid_wf/HYPE_240m_baseline.json --candidate-set hype-240
```

## A generic baseline through adapters

The dataset, the walk-forward windows and the metrics are shared in `offline/backtests/`.
The decision engine is not shared: each venue runs its own live strategy through a replay
adapter.

```text
canonical OHLC + hash
        │
        ▼
shared walk-forward evaluator
        │
        ├── Kraken adapter ──────► kraken/Strategy.step
        ├── Binance adapter ─────► binance_api/Strategy.step
        ├── Hyperliquid adapter ─► hyperliquid/Strategy.step
        └── Trading212 adapter ──► 212trading/Strategy.step
```

Kraken:

```bash
.venv/bin/python offline/runners/kraken_walk_forward_baseline.py
```

Trading212, using the profile's versioned configuration directly:

```bash
.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile nvda --range 2y --interval 1d

# SPCX has a live gate on Yahoo 5m bars; the replay refuses other cadences:
.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile spcx --range 1mo --interval 5m
```

The Trading212 runner does not touch the order API and cannot place trades. For non-USD
profiles it automatically downloads and freezes the historical FX series; your own CSV can be
supplied through `--fx-dataset`. `--fx-to-usd` is only a fixed override.

## Execution stress

The same options exist on both runners:

```bash
# HYPE: a conservative scenario, not a calibrated estimate of the real costs
.venv/bin/python offline/runners/kraken_walk_forward_baseline.py \
  --spread-bps 20 --market-slippage-bps 30 \
  --partial-fill-ratio 0.5 --intrabar-policy worst_case

# Trading212: the current orders are limit orders; the spread affects the touch,
# and the remaining tranche continues in the next bar
.venv/bin/python offline/runners/t212_walk_forward_baseline.py \
  --profile nvda --range 2y --interval 1d \
  --spread-bps 10 --partial-fill-ratio 0.5 \
  --intrabar-policy worst_case
```

The Kraken comparator reads the model from the baseline report and applies it identically to
every candidate and every fee scenario.

## The HYPE financial benchmark and the promotion gate

The golden checks whether the engine takes the same decisions after a refactor. The financial
benchmark separately measures the OOS return and risk of a fixed profile under two execution
scenarios. Do not update the golden in order to "accept" a candidate, and do not promote a
candidate merely because it preserves the golden.

```bash
.venv/bin/python offline/runners/kraken_financial_benchmark.py \
  --output offline/research/hype_dataset/financial_baseline_v1.json \
  --markdown chatgpt_agent_work/HYPE_FINANCIAL_BENCHMARK.md

# An exact reproduction of the versioned baseline
.venv/bin/python offline/runners/kraken_financial_benchmark.py \
  --verify offline/research/hype_dataset/financial_baseline_v1.json \
  --output /tmp/hype_financial_verify.json \
  --markdown /tmp/hype_financial_verify.md
```

For a candidate, `--params-report` reads a `strategy_params` object and `--compare-to`
attaches the verdict to the report. The gate can also be run on its own:

```bash
.venv/bin/python offline/runners/financial_promotion_gate.py \
  offline/research/hype_dataset/financial_baseline_v1.json \
  offline/results/hype_financial/candidate.json
```

The gate has two labelled paths. `RETURN` requires a mean advantage of at least `0.10pp`, at
least 10 active windows, more windows won than lost, an exact sign test with `p <= 0.10`, and
that the worst fold and the drawdown are preserved. `DEFENSIVE` requires a non-inferior
return, a median Calmar computed per fold that is at least 15% better, a preserved Sortino, a
worst drawdown smaller by at least `1pp`, a better CVaR, exposure that has not grown, and at
least 10 windows whose drawdown change is supported by the sign test. A candidate is eligible
through `RETURN` or `DEFENSIVE`, but it keeps its label and goes through shadow separately.
Ties are not evidence. The cost values are provisional until they are calibrated from real
Kraken fills.

The priority HYPE set (`tp4`, `dca15`, progressive spacing, volatility-based DCA sizing, A, B
and `overlay650t8`) is run as a batch, with no grid search and without changing the live
configuration:

```bash
.venv/bin/python offline/runners/kraken_financial_compare.py
```

The runner first reproduces the versioned live baseline and stops the comparison if it
differs. It then applies the promotion gate to every candidate in the central and stress
scenarios.

The real audit can be aggregated read-only before the costs are calibrated:

```bash
.venv/bin/python offline/runners/calibrate_execution_audit.py \
  logger/execution_audit --venue Kraken \
  --output /tmp/kraken_execution_calibration.json \
  --markdown /tmp/kraken_execution_calibration.md
```

The report measures fees, latency, partial fills, the deviation of LIMIT fills and, for new
MARKET orders, the total shortfall between the decision price and the fill. That shortfall
lumps together market movement, spread and slippage; the report does not claim it can
separate them without the bid/ask/mid saved at decision time.
