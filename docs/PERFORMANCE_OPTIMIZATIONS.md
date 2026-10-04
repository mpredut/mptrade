# Fleet Performance Optimizations Backlog

This document details the repetitive computational hotspots identified across the trading bot fleet, the architectural risks involved (e.g. WebSocket dropouts), and the recommended implementation strategies.

---

## 1. Hotspot 1: Polynomial Curve Fitting & Trend Statistics (`priceAnalysis.py` / `instant_trend.py`)

### Current State
- On every 1-second tick loop across active symbols (BTC, TAO, ARB, HYPE), the bot re-executes:
  - `np.polyfit()` (2nd and 3rd degree polynomial regressions) across historical price series.
  - Autocovariance matrix calculations for the **Hurst exponent**.
  - **Mann-Kendall monotonic trend test** over sliding windows of thousands of samples.
- Even when no new price ticks have arrived or when the time delta is only a few hundred milliseconds, the entire regression matrix is recomputed from scratch.

### Proposed Optimization
- **Tick-ID / Timestamp Memoization**: Cache the regression and statistical outcomes keyed by `(symbol, last_tick_timestamp, window_seconds)`.
- **Threshold-Driven Recalculation**: Only recompute regressions if at least $K$ new ticks have arrived ($K \ge 3$) or if an elapsed time threshold (e.g. 3.0 seconds) has expired.
- **Expected Impact**: Reduces CPU utilization across bot worker processes by **50% to 70%**.

---

## 2. Hotspot 2: Position Filtering & Cost-Basis Recalculation (`monitortrades.py`)

### Current State
- On every iteration of the main loop, `monitortrades.py` re-scans the full list of historical orders (`trade_orders_buy`, `trade_orders_sell`) to calculate active open positions and average cost basis.
- In 99.9% of iterations, no new order execution or fill has occurred on the exchange.

### Proposed Optimization & Risk Analysis
- **Dirty-Flag Pattern**: Only invalidate and recalculate position state when an execution event occurs (`ORDER_TRADE_UPDATE` via WebSocket or order status change via REST).
- **Critical Risk (WebSocket Drops)**: If the system relied *exclusively* on WebSocket push events, any network hiccup or dropped socket frame could leave the bot with a stale cost-basis or missed fill.
- **Safety Solution (Hybrid Dirty-Flag with Fallback Heartbeat)**:
  - Primary trigger: Event-driven `dirty = True` on WS execution report.
  - Safety fallback: Periodic heartbeat poll (e.g. every 10–15 seconds) forces a re-verification of open positions against the local order cache and REST poller.
- **Expected Impact**: Eliminates redundant disk/memory scans and stabilizes loop latency to under 2 ms.

---

## 3. Hotspot 3: Rolling Window Volatility & Log Returns (`tradeall.py` / `shadow_signals.py`)

### Current State
- Standard deviation and noise floor bands are computed by iterating over all historical bars in a 1-hour / 4-hour list ($O(N)$ iteration on every bar).

### Proposed Optimization
- **Welford's Algorithm / Sliding Sum-of-Squares**: Update mean and variance incrementally in $O(1)$ time complexity as new bars enter and old bars exit the sliding window.
- **Expected Impact**: Constant-time $O(1)$ variance tracking without iterating over thousands of data points.

---

## 4. Pillar 3 & 4 Real-Time Shadow Alerting (`order_guard.py` & `notify_engine`)

### Concept
- In `shadow` mode, guards currently output evaluations to stdout logs (`[GEMINI_GUARD_SHADOW]`, `[GEOPOLITICAL_GUARD_SHADOW]`).
- Enable live push alerts to the operator's phone via `ntfy.sh` (topic `NTFY_TOPIC_GUARD`) whenever:
  - Google Gemini evaluates a purchase $\ge 1,000$ EUR and recommends a veto or downscale.
  - Geopolitical / Energy shock threat analyzer flags `ELEVATED` or `CRITICAL_SHOCK` risk.
- **Safety**: Purely observational; does not alter order routing or block funds, but provides immediate real-world validation of AI reasoning.
