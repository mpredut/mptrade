# Chop Regime & Fallback Weight Benchmark (2026-10-05)

## Overview & Objective
Empirical backtest evaluating position sizing and trade allocation policies during sideways / consolidation regimes (when the Mann-Kendall test indicates no statistically significant trend, $p > 0.05$).

Conducted to address the parameterization of fallback weights when a symbol has no active long-term trend, comparing flat fractional weights (`0.00`, `0.01`, `0.02`, `0.03`, `0.05`, `0.10`) against the market beta proxy (`PROXY_BTC`).

---

## Dataset & Execution Parameters
* **Historical Span:** 403.9 days (August 2025 – October 2026).
* **Tick Volume:** ~902,698 ticks for `TAOUSDC`, ~902,158 ticks for `BTCUSDC`.
* **Resampling Resolution:** 15-minute bars (30,152 bars for TAO, 30,016 bars for BTC).
* **Account Baseline:** \$10,000 initial equity, \$1,000 standard notional tranche.
* **Execution Guard Layer:** 0.1% taker fees, +3.5% Take-Profit, -4.0% Risk Stop.

---

## Empirical Benchmark Results

### 1. Altcoins: TAOUSDC (403.9 days, 30,152 bars)

| Policy | Net Return | Net Profit | Max Drawdown | Win Rate | Profit Factor | Total Trades | Chop Trades | Chop PnL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CHOP_0.00 (SKIP)`** | +0.51% | +$51.48 | **1.96%** | 53.97% | 1.08 | 252 | 0 | $0.00 |
| **`CHOP_0.01 (1%)`** | +0.37% | +$37.49 | 2.07% | 54.52% | 1.07 | 343 | 104 | +$2.59 |
| **`CHOP_0.02 (2%)`** | +0.39% | +$39.04 | 2.06% | 54.52% | 1.07 | 343 | 104 | +$5.18 |
| **`CHOP_0.03 (BASE)`** | +0.41% | +$40.59 | 2.05% | 54.52% | 1.07 | 343 | 104 | +$7.77 |
| **`CHOP_0.05 (5%)`** | +0.44% | +$43.69 | 2.02% | 54.52% | 1.07 | 343 | 104 | +$12.95 |
| **`CHOP_0.10 (10%)`** | +0.51% | +$51.44 | **1.95%** | 54.52% | 1.08 | 343 | 104 | +$25.90 |
| **`PROXY_BTC`** | **+0.54%** | **+$53.73** | 2.31% | **54.52%** | **1.08** | 343 | 104 | **+$24.87** |

### 2. Market Anchor: BTCUSDC (403.9 days, 30,016 bars)

| Policy | Net Return | Net Profit | Max Drawdown | Win Rate | Profit Factor | Total Trades | Chop Trades | Chop PnL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CHOP_0.00 (SKIP)`** | -1.21% | -$121.23 | **1.53%** | **47.62%** | 0.70 | 63 | 0 | $0.00 |
| **`CHOP_0.01 (1%)`** | **-1.08%** | **-$108.37** | **1.50%** | 47.37% | 0.70 | 95 | 44 | -$2.55 |
| **`CHOP_0.02 (2%)`** | -1.11% | -$111.36 | 1.54% | 47.37% | 0.70 | 95 | 44 | -$5.09 |
| **`CHOP_0.03 (BASE)`** | -1.14% | -$114.34 | 1.57% | 47.37% | 0.70 | 95 | 44 | -$7.64 |
| **`CHOP_0.05 (5%)`** | -1.20% | -$120.32 | 1.65% | 47.37% | 0.70 | 95 | 44 | -$12.73 |
| **`CHOP_0.10 (10%)`** | -1.35% | -$135.24 | 1.83% | 47.37% | 0.71 | 95 | 44 | -$25.46 |

---

## Statistical Insights & Conclusions

1. **Altcoin Alpha via Proxy Trend (`PROXY_BTC`):**
   * For altcoins like `TAOUSDC`, using `PROXY_BTC` during local consolidation yields the **highest total return (+0.54%)** and net profit.
   * *Mechanism:* When an altcoin is lagging in chop while Bitcoin establishes an upward trend, allocating based on Bitcoin's trend weight provides valuable pre-breakout accumulation at favorable prices.

2. **Capital Preservation on Bitcoin:**
   * For the market anchor (`BTCUSDC`), active trading during sideways regimes incurs fee and spread drag (-$7.64 across 44 chop trades at 0.03).
   * Increasing chop weights to 0.05 or 0.10 exacerbates drawdown (rising from 1.50% to 1.83%).
   * Keeping the fallback weight low (`0.01`–`0.03`) prevents capital bleed while maintaining active order pipeline liveness.

3. **Optimal Configuration Enforced:**
   * **Altcoin Proxy:** `binance_weight_proxy = BTCUSDC` and `kraken_weight_proxy = BTCUSDC`.
   * **Conservative Fallback:** `default_chop_weight = 0.03`, `binance_chop_weight = 0.03`, and `kraken_chop_weight = 0.03`.
   * **Code Parity:** Unified in `order_guard.py` (`resolve_trade_weight` + `compute_weight_capped_qty`) and consumed identically by `bapi_placeorder.py` and `order_guard.weight_limit`.
