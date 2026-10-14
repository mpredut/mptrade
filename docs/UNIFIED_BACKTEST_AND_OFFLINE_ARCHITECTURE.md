# Unified Backtest & Offline Walk-Forward Architecture

This document serves as the persistent architectural specification and design reference for the unified backtesting, simulation, and walk-forward offline testing framework in `mptrade`.

---

## 1. Architectural Tenets & Design Principles

The backtesting subsystem was unified to eliminate logic drift between live trading and offline simulation:

1. **Single Source of Truth (SSOT) Strategy Logic:**
   - Both live trading and historical backtests execute the identical decision and financial logic in [`strategies/spot_engine.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/strategies/spot_engine.py) and [`strategies/spot_rules.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/strategies/spot_rules.py).
   - Parameters and thresholds are shared via [`StratParams`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/strategies/spot_engine.py). There is zero mathematical divergence between simulation decisions and production execution.

2. **Decoupled Simulation Engine (SoC):**
   - The strategy makes decisions (`step()`); the replay engine separately models fills, slippage, orderbooks, fee reserves, intrabar collisions, and multi-horizon trailing stops without network, external I/O, or persistent state.
   - Canonical replay simulation is centralized in [`strategies/spot_replay.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/strategies/spot_replay.py).

3. **Linear Dispatcher Choke Point:**
   - Instead of fragmented runner invocations, operators and automated CI invoke a single entry point: [`offline/runners/walk_forward.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward.py).
   - Candidate comparisons across all venues route through [`offline/runners/walk_forward_compare.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward_compare.py).

---

## 2. Multi-Venue Backtesting Matrix

All 4 production trading platforms share symmetric capabilities in the offline layer:

| Venue | Market Type | Configuration Source | Baseline Runner | Replay Adapter | Candidate Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kraken** | Spot Crypto | `.env` / `config.env` | [`kraken_walk_forward_baseline.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/kraken_walk_forward_baseline.py) | [`kraken/replay.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/kraken/replay.py) (SSOT shim) | [`walk_forward_compare.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward_compare.py) |
| **Binance** | Spot Crypto | `config.env` | [`binance_walk_forward_baseline.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/binance_walk_forward_baseline.py) | [`binance_api/replay.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/binance_api/replay.py) | [`walk_forward_compare.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward_compare.py) |
| **Hyperliquid** | Perpetual | `hyperliquid/config.env` | [`hyperliquid_walk_forward_baseline.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/hyperliquid_walk_forward_baseline.py) | [`hyperliquid/replay.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/hyperliquid/replay.py) | [`walk_forward_compare.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward_compare.py) |
| **Trading 212** | Equity CFDs | `212trading/config.<p>.env` | [`t212_walk_forward_baseline.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/t212_walk_forward_baseline.py) | [`212trading/replay.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/212trading/replay.py) | [`walk_forward_compare.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward_compare.py) |

---

## 3. The Canonical Replay Engine (`strategies/spot_replay.py`)

### Simulation Lifecycle & Execution Flow

```text
Historical OHLC Series (Frozen Dataset CSV)
                   │
                   ▼
       Warmup Bars (Shadow Prices & Surge Buffers)
                   │
                   ▼
  For each Bar: [Open, High, Low, Close]
    ├── 1. Intrabar Fill Evaluation (High / Low / Open)
    │      ├── Intrabar Stop-Loss Breaches (Adverse Extremes)
    │      ├── Limit DCA / Buy Entries (Touch Evaluation)
    │      └── Take-Profit / Trailing Pullback Exits
    ├── 2. Ambiguity Resolution (ExecutionModel: buy_first | sell_first | worst_case)
    ├── 3. Slippage & Spread Stress (bps adjustment)
    ├── 4. Order TTL & Runaway Price Cancellation (Live reconcile replica)
    ├── 5. Strategy Step (Evaluates indicators, state transitions, and new orders)
    └── 6. Account & Balance Accounting (Finite cash, reserve caps, fee deductions)
```

### Key Mechanical Realizations
- **Silent Execution:** Both `_strat.notify` and `_strat.log` are replaced with `_silent` during replay iterations (with `try/finally` restoration). This eliminated tens of thousands of stdout lines per sweep and reduced full test regression runtimes by over 30%.
- **Intrabar Ambiguity Resolution:** When a single candle's $[L, H]$ range touches both an open (DCA) and a close (TP) limit, the engine uses [`ExecutionModel`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/backtests/execution.py) policies rather than making optimistic fill assumptions.
- **Runaway & TTL Handling:** Emulates live `_reconcile_real()`: limit orders that expire past `order_ttl_min` or whose price runs away by $>0.3\%$ are cancelled so subsequent bars can place fresh entries.

---

## 4. Top-Level Unified CLI Dispatcher (`offline/runners/walk_forward.py`)

The unified CLI facade dispatches commands across all venues and modes cleanly:

### Usage Reference

```bash
# 1. Baseline Evaluation (Characterization across rolling out-of-sample folds):
python offline/runners/walk_forward.py --venue binance --symbol BTCUSDT --intervals 240
python offline/runners/walk_forward.py --venue kraken --intervals 240 --dataset 240=path/to/candles.csv
python offline/runners/walk_forward.py --venue hyperliquid --coin HYPE --direction long --intervals 240
python offline/runners/walk_forward.py --venue t212 --profile nvda --range 2y --interval 1d

# 2. Candidate One-Factor Robustness Comparison:
python offline/runners/walk_forward.py --mode compare offline/results/binance_wf/BTCUSDT_240m_baseline.json
python offline/runners/walk_forward.py --mode compare offline/results/kraken_wf/baseline_HYPEUSD_*.json --candidate-set hype-240
python offline/runners/walk_forward.py --mode compare offline/results/hyperliquid_wf/HYPE_240m_baseline.json --candidate-set hype-240
```

### Features
- **Schema-Agnostic Baseline Ingestion:** [`walk_forward_compare.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/walk_forward_compare.py) automatically ingests either multi-interval (Kraken) or single-interval (Binance, Hyperliquid) baseline JSON reports.
- **Venue Fee Stress Schedules:** Automatically stresses fee schedules per venue:
  - Binance: `0.05%, 0.075%, 0.10%, 0.15%`
  - Kraken: `0.16%, 0.26%, 0.35%, 0.50%`
  - Hyperliquid: `0.02%, 0.035%, 0.05%, 0.08%`
  - Trading 212: `0.00%, 0.02%, 0.05%, 0.10%`
- **Multi-Criteria Pareto Dominance:** Calculates Pareto frontiers across:
  1. Mean return percentage
  2. Worst-case segment return
  3. Worst-case maximum drawdown
  4. Win / Tie / Loss ratios against the live baseline configuration

---

## 5. Master Suite & CI Regression Architecture

- **Master Overnight Suite ([`offline/runners/run_overnight_master_suite.sh`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/run_overnight_master_suite.sh)):**
  - Uses the unified `walk_forward.py` facade for Steps 1A, 1B, 2A, 2B, 2C, 2D, and 2E.
- **Targeted Financial Baseline ([`offline/runners/run_financial_baseline.sh`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/run_financial_baseline.sh)):**
  - Runs targeted unit tests covering all backtest adapters, replay rules, Pareto dominance calculations, and dispatcher facades in under 3 minutes.
- **Asynchronous DEV CI Regression:**
  - Automated via [`offline/runners/trigger_dev_regression.sh`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/trigger_dev_regression.sh) and monitored via [`check_dev_regression.sh`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/runners/check_dev_regression.sh).
  - Status verified at **1446 passed tests, 0 failures** in **99s**.

---

## 6. Conclusions & Operating Guidelines

1. **Refactoring Milestone Complete:** The offline testing and backtest harness requires no further structural changes. Any future strategy variations or ablations should be added as candidate definitions in [`offline/backtests/hype_candidates.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/offline/backtests/hype_candidates.py) or `walk_forward_compare.py`.
2. **Strict Single Source of Truth:** Never implement custom replay or mathematical simulation code inside runner scripts. All simulation logic belongs strictly inside [`strategies/spot_replay.py`](file:///wsl.localhost/Ubuntu-24.04/home/mariusp/mptrade/strategies/spot_replay.py).
3. **Execution Safety:** When running multi-factor sweeps, use frozen CSV datasets with verified SHA256 hashes to guarantee reproducibility.
