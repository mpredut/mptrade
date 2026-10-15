# MPTrade Fortress Architecture: Pillars, Guards, Brakes & Triggers

**Status:** Production Reference  
**Last Updated:** October 2026  
**Audience:** Core Developers, Quant Officers, System Architects  

---

## 1. Top-Level Entry Point & Architectural Choke Point

[`order_guard.py`](file:///home/predut/mptrade/order_guard.py) is the **single top-level root entry point** for all trading engines (`tradeall`, `rtrade`, `monitortrades`, `kraken`, `hyperliquid`). No order can reach an exchange without traversing this choke point.

`order_guard.py` coordinates the **Three-Pillar Fortress** in a strict sequential pipeline, maintaining clean separation of concerns and deterministic fail-closed protection:

```mermaid
flowchart TD
    Order["🎯 NEW ORDER INTENT (BUY / SELL)\nChoke Point: order_guard.py"] --> P1["🛡 PILLAR 1: internal_order_guard.py\nMathematical & Account Invariants\n⚡ 0 ms | 🚫 NO LLM"]
    
    P1 -- Invariant Violated --> VetoP1["⛔ VETO_STOP (Immediate Rejection)"]
    P1 -- Passed --> P2["🛡 PILLAR 2: external_order_guard.py\nMicrostructure: Orderbook, Funding, Whales\n⚡ 0 ms local cache | 🚫 NO LLM"]
    
    P2 -- Massive Ask Wall / Severe Imbalance --> DeferP2["⏳ DEFER_WAIT (Defer in worker retry buffer)"]
    P2 -- Crowded Perp Funding --> ScaleP2["📉 DOWNSCALE_QTY (Scale: 50%)"]
    P2 -- Passed / Scaled --> P3["🛡 PILLAR 3: intelligence_order_guard.py\nAI & LLM Intelligence Layer\nGeopolitical Shield + High-Stake Vetting"]
    
    P3 -- War / Energy Critical Shock --> VetoGeo["⛔ VETO_STOP (P3A · GEO-SHIELD)"]
    P3 -- Elevated Geopolitical Risk --> ScaleGeo["📉 DOWNSCALE_QTY (P3A · GEO-SHIELD: 25-50%)"]
    P3 -- Large Order (≥ €1k) Vetoed by LLM --> VetoStake["⛔ VETO_STOP (P3C · HIGH-STAKE)"]
    P3 -- Large Order (≥ €1k) Scaled by LLM --> ScaleStake["📉 DOWNSCALE_QTY (P3C · HIGH-STAKE)"]
    P3 -- Passed / Approved --> Exec["🚀 EXCHANGE DISPATCH\n(Executed with Final Safe Quantity)"]
```

---

## 2. LLM Usage Classification Matrix

To ensure absolute transparency between microsecond execution and qualitative reasoning, LLM usage is strictly classified across the three pillars:

| Pillar | Guard Module | LLM Status | Execution Latency | Frequency / Cadence |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1** | [`internal_order_guard.py`](file:///home/predut/mptrade/internal_order_guard.py) | 🚫 **NO LLM** (Pure Math) | $0\text{ ms}$ (In-memory) | Every single order intent |
| **Pillar 2** | [`external_order_guard.py`](file:///home/predut/mptrade/external_order_guard.py) | 🚫 **NO LLM** (Pure Microstructure) | $0\text{ ms}$ (Local cache) | Every BUY order intent |
| **Pillar 3** | [`intelligence_order_guard.py`](file:///home/predut/mptrade/intelligence_order_guard.py)<br>↳ [`geopolitical_order_guard.py`](file:///home/predut/mptrade/geopolitical_order_guard.py) | 🤖 **BACKGROUND LLM** | $< 1\text{ ms}$ at pre-trade | Asynchronous cycle: **Every 4.5 hours** |
| **Pillar 3** | [`macro_analyzer.py`](file:///home/predut/mptrade/intelligence/macro_analyzer.py)<br>↳ `sentiment_advisor_eval.json` | 🤖 **BACKGROUND LLM** | $0\text{ ms}$ at pre-trade | Asynchronous cycle: **Every 3.0 hours** |
| **Pillar 3** | [`intelligence_order_guard.py`](file:///home/predut/mptrade/intelligence_order_guard.py)<br>↳ [`high_stake_guard.py`](file:///home/predut/mptrade/intelligence/sentiment/guards/high_stake_guard.py) | 🤖 **ACTIVE LLM** | $\sim 10\text{--}15\text{ s}$ | Event-driven: **Only on BUY $\ge 1,000\text{ EUR}$** |

---

## 3. Pillar-by-Pillar Detailed Architecture

### Pillar 1: Mathematical Invariants & Base Risk (Internal)
* **File:** [`internal_order_guard.py`](file:///home/predut/mptrade/internal_order_guard.py)
* **Role:** Internal mathematical invariants, capital preservation, execution mechanics.
* **LLM:** 🚫 **NO LLM** — Deterministic math, $0\text{ ms}$.
* **Guards & Triggers:**
  1. `daily_limit_guard`: Rolling 24h loss budget (`max_daily_loss_eur`). Blocks trading if account drawdown threshold is breached.
  2. `profit_guard` & `surge_profit_floor_guard`: Reentry profit floor & trailing margin protection. Prevents selling below breakeven basis or dumping into an active parabolic surge.
  3. `inventory_guard`: Position exposure caps per symbol and globally.
  4. `stale_cache_guard` & `spread_and_liquidity_guard`: Rejects orders on stale price ticks ($> 10\text{s}$) or abnormally wide spreads.
  5. `weibull_exhaustion_guard`: Statistical Weibull hazard ratio detecting trend exhaustion. Triggers `DOWNSCALE_QTY` (default: 25% scale via `weibull_exhausted_scale = 0.25`).

---

### Pillar 2: Market Microstructure & Flow (External)
* **File:** [`external_order_guard.py`](file:///home/predut/mptrade/external_order_guard.py) *(mirrors `internal_order_guard.py`)*
* **Role:** Spot orderbook depth, large whale limit walls, perpetual derivatives crowding.
* **LLM:** 🚫 **NO LLM** — Pure quantitative market microstructure, $0\text{ ms}$ local telemetry cache.
* **Guards & Triggers:**
  1. `OrderbookWallGuard` (`[P2 · ORDERBOOK]`):
     - *Telemetry:* `cachedb/orderbook_depth_collect.json`
     - *Trigger 1:* `min_buy_imbalance < 0.25` (less than 25% bids in top 20 book levels).
     - *Trigger 2:* Overhead ask wall exceeding `whale_wall_usd_limit` (default: $\$1,000,000$) within 1–2% of entry price.
     - *Brake:* `DEFER_WAIT` (holds order in worker retry queue until wall softens or pulls) or `VETO_STOP`.
  2. `FundingCrowdingGuard` (`[P2 · FUNDING]`):
     - *Telemetry:* `cachedb/funding_rates_collect.json`
     - *Trigger:* Perpetual funding rate exceeds `funding_max_long_rate = 0.0005` (0.05% per 8h), signaling overcrowded long positioning prone to long squeezes.
     - *Brake:* `DOWNSCALE_QTY` (scales down size by `funding_crowded_scale = 0.50`) or `VETO_STOP`.
  3. `WhaleDivergenceGuard` (`[P2 · WHALE-FLOW]`):
     - *Telemetry:* `cachedb/whale_snapshot_<SYM>.json` (correlating $\Delta P_{1h}$, $\Delta OI_{1h}$, Taker Buy/Sell ratio, and perpetual funding rate).
     - *Trigger 1 (OI Divergence & Regime Matrix):* Classifies 8 distinct microstructural regimes:
       - `aggressive_shorting`: Heavy short expansion ($P \downarrow, OI \uparrow, \text{Taker} < 0.85$). *Brake:* `DEFER_WAIT` on BUY.
       - `short_squeeze_primed`: Shorts trapped underwater ($P \uparrow, OI \uparrow$ with negative funding or crowded shorts). *Brake:* `DEFER_WAIT` on SELL (allows rally to mature); permits BUY.
       - `short_squeeze`: Active forced liquidation cascade ($P \uparrow, OI \downarrow, \text{Taker} \ge 1.25$). *Brake:* Permits BUY momentum; `DEFER_WAIT` on SELL.
       - `short_covering`: Passive low-volume fakeout ($P \uparrow, OI \downarrow, \text{Taker} < 1.25$). *Brake:* `DEFER_WAIT` on BUY (`short_covering_fakeout`).
       - `long_liquidation`: Involuntary forced liquidation cascade ($\Delta OI \le -3.0\%$, taker dump $< 0.85$). *Brake:* `DEFER_WAIT` on BUY (`liquidation_cascade_active`).
       - `long_unwinding`: Orderly voluntary position closing (gentle OI drop, healthy taker ratio). *Brake:* Permits BUY with progressive downscaling (`DOWNSCALE_QTY`) only if taker selling is noticeable.
     - *Trigger 2 (Whale Taker Flow Supremacy):* Evaluates aggressive market taker sells against depth (`min_taker_ratio = 0.75`). If taker sell volume ratio breaches `taker_downscale_threshold = 0.90`, it triggers progressive size downscaling (`taker_downscaled_scale = 0.50`, `DOWNSCALE_QTY`) or absolute `VETO_STOP` before entering falling knives.
  4. `SmartStopLossGuard` (`[P2 · SMART-SL]`):
     - *Anti-Wick Hunt & Squeeze Protection:* Recognizes `short_squeeze_primed` and `short_squeeze` alongside whale long positioning ($\ge 60\%$) and book absorption, granting a 120s grace period instead of prematurely selling out of predatory wick hunts.

---

### Pillar 3: AI Intelligence Layer (Geopolitical Shield, Sentiment Advisor & High-Stake LLM)
* **Files:** [`intelligence_order_guard.py`](file:///home/predut/mptrade/intelligence_order_guard.py), [`geopolitical_order_guard.py`](file:///home/predut/mptrade/geopolitical_order_guard.py), [`intelligence/sentiment/sentiment_advisor.py`](file:///home/predut/mptrade/intelligence/sentiment/sentiment_advisor.py), [`intelligence/sentiment/guards/high_stake_guard.py`](file:///home/predut/mptrade/intelligence/sentiment/guards/high_stake_guard.py), [`intelligence/macro_analyzer.py`](file:///home/predut/mptrade/intelligence/macro_analyzer.py)
* **Role:** Qualitative AI reasoning, macro shock detection, and real-time high-notional risk vetting.
* **LLM:** 🤖 **ACTIVE & ASYNC LLM** (Google Gemini via isolated `agy` CLI).
* **Guards & Sub-Components:**
  1. `GeopoliticalThreatShield` (`[P3A · GEO-SHIELD]`):
     - *Async Background Cycle:* Evaluated every 4.5 hours (`macro_llm_interval_h = 4.5`) via LLM reasoning on international news feeds (`cachedb/news_feed_collect.json`).
     - *Output File:* `cachedb/geopolitical_threat_eval.json`.
     - *Pre-Trade Check:* $< 1\text{ ms}$ local disk cache read on every BUY order.
     - *Threat Levels:*
       - `NORMAL` ($\text{risk} < 0.35$): `ALLOW`.
       - `ELEVATED` ($0.35 \le \text{risk} < 0.70$): `DOWNSCALE_QTY` (scale to 50%).
       - `HIGH` ($0.70 \le \text{risk} < 0.85$): `DOWNSCALE_QTY` (scale to 25%).
       - `CRITICAL_SHOCK` ($\text{risk} \ge 0.85$, kinetic war escalation, oil embargo, systemic banking freeze): `VETO_STOP` (hard block on all new long entries).
  2. `SentimentAdvisor` (`[P3B · SENTIMENT]`):
     - *Async Trigger:* Periodic background cadence every 3.0 hours (`sentiment_advisor_interval_h = 3.0`).
     - *Telemetry:* `cachedb/fear_greed_collect.json`, 24h market breadth (advancers/decliners), top trader long/short exposure.
     - *Output File:* `cachedb/sentiment_advisor_eval.json`.
     - *Regimes:*
       - **Market Bias:** `BULLISH` | `NEUTRAL` | `CAUTION` | `BEARISH`
       - **Recommended Action:** `ALLOW` | `TRIM_PROFITS` | `DEFENSIVE` | `HALT_NEW_BUYS`
       - **Risk Level:** `LOW` | `MODERATE` | `HIGH` | `EXTREME`
  3. `HighStakeGuard` (`[P3C · HIGH-STAKE]`):
     - *Pre-Trade Trigger:* Fires **for orders $\ge$ `llm_min_notional_eur = 400.0`**. Orders below 400 EUR pass immediately with $0\text{ ms}$ latency (`ACCEPT`).
     - *Context Ingested:* Real-time synthesis of L2 orderbook imbalance, overhead whale walls, funding rates, whale OI positioning, Fear & Greed index, and current geopolitical threat state.
     - *Decisions:*
       - `APPROVED`: Allows execution (`ALLOW`, 100% scale).
       - `DOWNSCALE`: Reduces size (`DOWNSCALE_QTY`, suggested scale 25%–75%).
       - `REJECTED`: Vetoes order (`VETO_STOP`, 0% scale).
     - *Fallback Policy:* Configurable via `llm_fallback = allow | downscale | veto` on timeout (`llm_timeout_sec = 25.0`).

---

## 4. The Unified Brake System (`BrakeAction`)

Canonical definition in [`intelligence/internal/guards/guard_decision.py`](file:///home/predut/mptrade/intelligence/internal/guards/guard_decision.py):

| Brake Action | Execution Effect | Scale Range | Notification Label |
| :--- | :--- | :--- | :--- |
| **`ALLOW`** | Order proceeds unaltered | `1.0` (100%) | `🧭 ACCEPT` |
| **`DOWNSCALE_QTY`** | Scales down notional / quantity | `0.10` – `0.75` | `🛡 SCALE {pct}%` |
| **`DEFER_WAIT`** | Defers execution in worker retry queue | `0.0` (transient) | `🛡 DEFER` |
| **`VETO_STOP`** | Immediate rejection of order intent | `0.0` (0%) | `🛡 BLOCK` |
| **`HALT_SYSTEM`** | Emergency kill switch (halts all engine activity) | `0.0` | `🛑 EMERGENCY HALT` |

### Guard Operation Modes ([`order_guard.conf`](file:///home/predut/mptrade/order_guard.conf))
* **`off`**: Guard is bypassed completely (zero evaluations).
* **`shadow`**: Guard runs full analytics, logs actions, and dispatches push alerts on phone (`shadow_notify = 1`), but **does not alter or block** real orders.
* **`enforce`**: Guard actively intervenes in live order routing (vetoes, scales, or defers).

### Composition & Priority Rules
1. **Veto Precedence (Fail-Closed):** If any guard (P1, P2, or P3) returns `VETO_STOP`, the order is blocked immediately.
2. **Compound Downscaling:** Multiple downscale recommendations compound conservatively:
   $$\text{Final Scale} = \min(\text{Scale}_{\text{P1}}, \text{Scale}_{\text{P2}}, \text{Scale}_{\text{P3}})$$
3. **Transient Deferral:** `DEFER_WAIT` routes the order into [`order_retry.py`](file:///home/predut/mptrade/order_retry.py), re-checking micro conditions on each cycle rather than discarding intent permanently.

---

## 5. Telemetry & Cache Storage Standard (`cachedb/`)

To decouple heavy network ingestion from microsecond trading loops, cache files follow a strict Single Source of Truth (SSOT) convention:
* **`*_collect.json`**: Raw telemetry collected from external feeds (orderbooks, rates, feeds).
* **`*_eval.json`**: Evaluated analytical synthesis produced by intelligence engines.

```text
cachedb/
├── orderbook_depth_collect.json     # P2: Spot top-20 bids/asks depth & wall detection
├── funding_rates_collect.json       # P2: Binance/Perp funding rate telemetry
├── whale_oi_collect.json            # P2: Open interest & whale net position telemetry
├── fear_greed_collect.json          # P3: Fear & Greed index (0-100)
├── news_feed_collect.json           # P3: Raw multi-source international news headlines
├── sentiment_advisor_eval.json      # P3: Sentiment Advisor evaluation (3.0h LLM cycle)
└── geopolitical_threat_eval.json    # P3: Geopolitical threat assessment (4.5h LLM cycle)
```

---

## 6. Notification & Alerting Engine ([`notify_engine/`](file:///home/predut/mptrade/notify_engine/))

All bot and guard alerts flow through `notify_engine.alertnotifiers.notify` into dedicated ntfy topics with deduplication and email mirroring:

| Category | Source Key | Ntfy Topic | Purpose & Policy |
| :--- | :--- | :--- | :--- |
| **GUARD** | `order_guard` | `ntfy-guard-1978` | P2 (Orderbook, Funding, Whale) and P3 High-Stake alerts. 30m per-symbol cooldown. |
| **MACRO** | `macro_shadow` | `ntfy-macro-1978` | P3 Sentiment Advisor regimes and P3 Geopolitical Threat changes. |
| **TRADES** | `tradeall`, `monitortrades` | `ntfy-trades-1978` | Live executions, position scaling, take-profit triggers. |
| **PRICE** | `price_notifier` | `ntfy-price-1978` | Key price threshold breakouts and trend reversals. |
| **ERROR** | `watchdog`, OS scripts | `ntfy-error-1978` | Critical system errors, auto-mirrored to SMTP Email via [`mailer.py`](file:///home/predut/mptrade/notify_engine/mailer.py). |
| **SERVER** | `orchestrator`, `deadman` | `ntfy-server-1978` | Daemon lifecycle, config reload notifications, and deadman/heartbeat monitoring. |

### Standard Alert Format
* **Title:** Direct and action-oriented: `🛡 [{PILLAR} · {GUARD}] {ACTION} {SIDE} {SYMBOL}`
* **Line 1 (Order Parameters):** `Order: {SIDE} {QTY} {SYMBOL} @ ${PRICE} · €{NOTIONAL} (Scale: {SCALE}%)`
* **Line 2 (Secondary Metrics):** `Imbalance: {BIDS}% bids vs {ASKS}% asks | Overhead Wall: ${WALL} @ ${PRICE}`
* **Line 3 (Concise Rationale):** `Reason: {1-2 sentence clean explanation without redundant prefixes}`

---

## 7. Antigravity OS Integration & Housekeeping

* **Dual Workspace Isolation:**
  - `mptrade` (`3ed4aa5d-...`): Protected interactive user engineering workspace.
  - `autonommptrade` (`f5f9d01f-...`): Sandboxed background bot workspace for automated LLM queries.
* **Protobuf Hub Synchronization:**
  - `ensure_proto_synced` in [`orchestratorOS/admin/prune_llm_threads.py`](file:///home/predut/mptrade/orchestratorOS/admin/prune_llm_threads.py) continuously mirrors entries from headless `jetbox_summaries_proto.pb` into GUI `agyhub_summaries_proto.pb`, ensuring all autonomous guard threads appear instantly in the Antigravity IDE sidebar.
* **Automated Retention Management:**
  - Automated autonomous threads older than 24 hours are pruned daily via cron, while manual engineering discussions in `mptrade` are strictly preserved.

---

## 8. In-Trade Fortress: `PositionExitManager` (MHDSL & MDHTP Mirror State Machine)

While `order_guard.py` acts as the **Pre-Trade Fortress** (vetoing or scaling incoming order intents), [`strategies/position_exit_manager.py`](file:///home/predut/mptrade/strategies/position_exit_manager.py) acts as the **In-Trade Fortress** across all fleet engines (`strategies/spot_engine.py`, `monitortrades.py`, and `212trading/strategy.py`).

`PositionExitManager` implements a mathematically symmetrical state machine governing position exits and post-exit re-entries:

```mermaid
flowchart TD
    subgraph SMMirror ["PositionExitManager: Mathematical Mirror State Machine"]
        direction TB
        subgraph MDHTP ["Take-Profit (MDHTP)"]
            TP0["Priority 0: Hard TP Spike\n(Partial Sell e.g. +20%)"] --> TP1["Priority 1: Trend Riding\n(Trend UP: Hold & Track Peak)"]
            TP1 --> TP2["Priority 2: Guaranteed Profit Floor\n(Protect min profit floor e.g. 0.8%)"]
            TP2 --> TP3["Priority 3: Dynamic Micro-Trailing\n(k=1.5 Volatility Pullback from Peak)"]
            TP3 --> TP4["Priority 4: Stagnation Timeout\n(18m without fresh peak)"]
            TP4 --> TP5["Priority 5: Trend Reversal\n(Trend turns down -> Market Exit)"]
        end

        subgraph MHDSL ["Stop-Loss (MHDSL)"]
            SL1["Priority 1: Hard Disaster Floor\n(max(ratio, vol_1h * 0.85, 0.8%), capped 15%)\nMonotonic Ratchet: never loosens"] --> SL2["Priority 2: Bounce Protection\n(Trend UP: Defer up to 220s)"]
            SL2 --> SL2A["Structural Breakdown\n(Break below prior swing low)"]
            SL2 --> SL2B["Bounce Rollover\n(Dynamic Ratchet on Pullback)"]
            SL2 --> SL2C["Bounce Timeout\n(220s elapsed without recovery)"]
            SL2 --> SL3["Priority 3: Smart SL Controller\n(Whale Absorption Filter)"]
        end
    end

    subgraph PostExit ["Symmetrical Re-Entry Mirror (MHDRE)"]
        MDHTP -->|Post-TP / Sale| RE_TP["TP Re-entry Mirror\n• Bull: Waits for healthy pullback from peak\n• Range: Requires discount drop below last sale"]
        MHDSL -->|Post-STOP| RE_SL["SL Re-entry Mirror\n• Tracks trough low\n• Confirmed recovery bounce above trough\n• Anti-Rollover Ratchet & Bear Trend Gate"]
    end
```

### Mathematical Symmetry Matrix

| Dimension | Take-Profit (MDHTP) | Stop-Loss (MHDSL) | Re-Entry Mirror (MHDRE) |
| :--- | :--- | :--- | :--- |
| **Zone Trigger** | $\text{Gain} \ge \text{gain\_threshold}$ | $\text{Loss} \ge \text{lost\_threshold}$ | Post-sale price evaluation |
| **Circuit Breaker** | Hard TP spike partial sell & guaranteed profit floor | Hard disaster floor ($\sigma_{1h}$ scaled, capped 15%) bypassing all deferrals | Stale barrier TTL expiration |
| **Trend Deferral** | Ride uptrend without premature exit | Defer stop on active uptrend, watch recovery | In bull trend, wait for pullback from peak |
| **Micro-Trailing** | Pullback from peak ($k=1.5 \times \text{vol}_{1h}$) | Bounce rollover from recovery peak | Anti-rollover ratchet on post-stop recovery bounce |
| **Dynamic Ratchet** | Tightens trail as profit expands | Tightens allowed pullback as bounce recovery expands (Monotonic Floor) | Rollover clamp prevents re-entry if bounce fails |
| **Timeout Guard** | 18-minute (1080s) stagnation timeout | 220-second bounce watch timeout | TTL decay on last sale barrier |

### Strict Isolation & Non-Interference
1. **Zero Cross-Contamination:** Stop-Loss evaluation and Take-Profit evaluation use isolated in-memory state dictionaries (`sl_bounce_state` vs `tp_ride_state`).
2. **Sequential Priority Gate:** In profit or neutral territory, Stop-Loss triggers are unreachable ($P_{\text{loss}} \le 0$). Take-Profit exits run autonomously with proven volatility trailing.
3. **Atomic Rollback:** If Stop-Loss triggers an emergency liquidation in `monitortrades.py`, open resting TP limit orders are preserved if placement is deferred by trend, and restored atomically if placement fails.
4. **Capped Disaster Buffer & Monotonic Floor:** `sl_disaster_buffer_max_pct = 0.15` (15% decimal) caps disaster buffer expansion under extreme volatility spikes. Once armed, the disaster floor operates monotonically and never loosens upward in loss.
5. **SSOT Volatility Normalization:** `intelligence.internal.state.volatility` standardizes ratio vs percentage conversions across all exit engines, preventing scaling mismatches.

