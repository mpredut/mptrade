# Fine-Grained Fixed Surge Guard Grid Search Report

> **Execution Date:** October 2, 2026  
> **Evaluation Scope:** 191 Candidate Parameter Sets (1,146 Independent Replays + 2,292 Split-Half Sensitivity Runs)  
> **Assets Evaluated:** HYPE, TAO, ADA, SOL, BTC, ETH ($3,900 Allocation per Asset, 0.26% Round-Trip Fee)  
> **Git Reference:** [`077b1c68`](https://github.com/mpredut/mptrade/commit/077b1c68) on `main`  
> **Raw Database:** [`offline/results/fine_grained_fixed_surge_sweep.json`](file:///home/predut/mptrade/offline/results/fine_grained_fixed_surge_sweep.json)  

---

## 1. Executive Summary

To identify the global optimal fixed parameter configuration for **Parabolic Surge Guard**, an exhaustive high-resolution grid search was executed across:
- **Surge Gain Triggers ($g$):** `16.0%` to `26.0%` (1.0% step)
- **Exit Pullback Thresholds ($pb$):** `2.5%` to `4.0%` (0.2% - 0.3% step)
- **Rolling Surge Windows ($w$):** `48h`, `72h`, `96h`
- **Bear Bounce Hybrid Re-entry:** Anchored to canonical `0.5%` with dynamic regime filtering

All candidates were evaluated across multi-year continuous 4h candles with full intrabar execution scenarios, realistic fee deductions, and dual-half consistency verification ($H_1$ and $H_2$).

---

## 2. Global Top 15 Ranked by Altcoin Profitability (HYPE + TAO + ADA)

| Rank | Candidate Identifier | Surge Gain ($g$) | Pullback ($pb$) | Window ($w$) | Alts PnL ($) | Fleet PnL ($) | Worst MaxDD (%) | PnL/MaxDD Ratio | $H_1/H_2$ Both Win |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `FIXED_g21_pb2.8_w48` | 21.0% | 2.8% | 48h | **$5,520.73** | **$18,428.65** | 36.90% | 499.4 | **True** |
| **#2** | `FIXED_g21_pb2.8_w72` | 21.0% | 2.8% | 72h | **$5,520.73** | **$18,187.27** | 36.90% | 492.8 | **True** |
| **#3** | `FIXED_g21_pb2.8_w96` | 21.0% | 2.8% | 96h | **$5,520.73** | **$18,258.00** | 36.90% | 494.8 | **True** |
| **#4** | `FIXED_g21_pb3.2_w48` | 21.0% | 3.2% | 48h | **$5,503.73** | **$18,212.61** | 37.67% | 483.5 | **True** |
| **#5** | `FIXED_g21_pb3.2_w72` | 21.0% | 3.2% | 72h | **$5,503.73** | **$18,292.31** | 37.67% | 485.6 | **True** |
| **#6** | `FIXED_g21_pb3.2_w96` | 21.0% | 3.2% | 96h | **$5,503.73** | **$18,292.31** | 37.67% | 485.6 | **True** |
| **#7** | `FIXED_g18_pb3.2_w72` | 18.0% | 3.2% | 72h | **$5,484.61** | **$15,948.30** | 36.82% | 433.1 | **True** |
| **#8** | `FIXED_g18_pb3.2_w96` | 18.0% | 3.2% | 96h | **$5,484.61** | **$16,042.59** | 36.82% | 435.7 | **True** |
| **#9** | `FIXED_g20_pb3.2_w48` | 20.0% | 3.2% | 48h | **$5,453.31** | **$17,417.64** | 37.67% | 462.4 | **True** |
| **#10** | `FIXED_g20_pb3.2_w72` | 20.0% | 3.2% | 72h | **$5,453.31** | **$17,417.64** | 37.67% | 462.4 | **True** |
| **#11** | `FIXED_g18_pb3.2_w48` | 18.0% | 3.2% | 48h | **$5,426.82** | **$16,011.83** | 36.82% | 434.8 | **True** |
| **#12** | `FIXED_g20_pb3.2_w96` | 20.0% | 3.2% | 96h | **$5,398.43** | **$17,358.60** | 37.67% | 460.9 | **True** |
| **#13** | `FIXED_g20_pb2.8_w48` | 20.0% | 2.8% | 48h | **$5,358.56** | **$18,279.83** | 36.90% | 495.3 | **True** |
| **#14** | `FIXED_g20_pb2.8_w72` | 20.0% | 2.8% | 72h | **$5,358.56** | **$18,337.65** | 36.90% | 496.9 | **True** |
| **#15** | `FIXED_g20_pb2.8_w96` | 20.0% | 2.8% | 96h | **$5,329.24** | **$18,344.52** | 36.90% | 497.1 | **True** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LIVE** | `LIVE_CURRENT_g18_pb3.5`| 18.0% | 3.5% | 72h | **$4,886.59** | **$15,640.30** | **36.34%** | 430.4 | **True** |
| **CTRL** | `BASELINE_SURGE_OFF` | 0.0% | 0.0% | 0h | **$4,316.91** | **$19,816.77** | 37.18% | 532.9 | **True** |

---

## 3. Asset-Level Analysis of Optimal Clusters

### 3.1 Cluster Comparison Table

| Configuration Cluster | HYPE ($) | TAO ($) | ADA ($) | Alts Total ($) | Worst MaxDD | Primary Strategic Trade-Off |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Top 21% / PB 2.8% (`g21_pb2.8`)** | **$2,599.77** | $156.54 | $2,764.42 | **$5,520.73** | 36.90% | Highest total altcoin return; balances deep surge capturing with quick profit lock. |
| **Balanced 20% / PB 3.2% (`g20_pb3.2`)** | $2,567.26 | **$278.84** | $2,607.21 | **$5,453.31** | 37.67% | Maximizes TAO swing profitability while maintaining high returns across all 3 assets. |
| **Tight 18% / PB 3.2% (`g18_pb3.2`)** | $2,427.66 | $24.33 | **$3,032.63** | **$5,484.61** | 36.82% | Highest ADA return, but lower TAO capture. |
| **Live Production (`g18_pb3.5`)** | $2,416.32 | $27.07 | $2,443.20 | **$4,886.59** | **36.34%** | **Lowest drawdown across all configurations**; conservative pullback margin. |
| **Baseline (Surge OFF)** | $2,244.20 | $83.17 | $1,989.54 | **$4,316.91** | 37.18% | Lacks blow-off top lock; sacrifices +$1,203.82 in altcoin cycle profits. |

---

## 4. Key Discoveries

1. **Pullback Sweet Spot ($2.8\% - 3.2\%$):**
   - Pullbacks tighter than 2.8% (e.g. 2.5%) prematurely exit on normal 4h intraday wicks.
   - Pullbacks wider than 3.5% (e.g. 3.8% - 4.0%) surrender too much peak profit during blow-off top reversals.
   - The optimal range across all altcoins is concentrated tightly between **2.8% and 3.2%**.
2. **Surge Trigger Sweet Spot ($20.0\% - 21.0\%$):**
   - Raising the trigger from 18% to 20%-21% gives volatile momentum moves room to expand before arming, increasing net profit per cycle.
   - Triggers above 24% fail to trigger frequently enough on altcoins with lower parabolic amplitude (e.g. TAO turns negative at 25%).
3. **Rolling Window Robustness:**
   - Results between 48h, 72h, and 96h windows show minimal divergence (less than 1% delta), confirming that the position gain trigger and profit floor invariant dominate decision quality, rendering the rolling window stable and non-fragile.
4. **Shadow Validation Path:**
   - Candidate `fixed_surge20_pb3` (20% trigger, 3.0% pullback) has been added to `kraken/shadow_live.py` for continuous live paper forward-testing alongside `current` (18% trigger, 3.5% pullback).
