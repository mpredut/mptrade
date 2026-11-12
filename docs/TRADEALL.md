# Tradeall policy and operational boundaries

`tradeall.py` consumes timestamped price ticks across configured venues (`binance`,
`kraken`, `hyperliquid`, `t212`) and evaluates two trend windows at sub-100ms latency.
`TrendCoordinator` subscribes to `CacheCurrentPriceManager` (`MODE_LIVE_READER`) and
listens on the non-blocking `AF_UNIX` datagram fanout (`cachedb/prices/current_price_ticks.sock`)
emitted by `cacheManager`, without opening duplicate venue WebSockets. It evaluates a
symbol as soon as it is dirty and `min_eval_interval_sec` (`0.05s` default via
`TRADEALL_MIN_EVAL_INTERVAL_SEC`) has elapsed, scheduling exact wakeup timeouts so ticks
arriving inside the debounce window are never delayed until the heartbeat ceiling.
Duplicate configured symbols are collapsed so one tick cannot subscribe or evaluate the
same market twice.

## Decision pipeline

1. `CacheCurrentPriceManager` + `current_price_ticks.sock` supply a timestamped quote.
   Closed-session markets (`not is_symbol_market_open`, e.g. US equities outside RTH) and
   stale, future, missing, non-finite or non-positive prices cannot reach signal evaluation
   or execution.
2. `handle_symbol` calculates short/big-window slope, gradient and position fields, reading
   `vol_1h_pct` from `24trend/live_<SYM>.json` via `(mtime_ns, size)` stat memoization so
   unchanged disk files are not re-parsed on every tick.
3. `logic` maintains an UP/DOWN `TrendState`, confirmation count, freshness and uniformity.
   Confirmation increments are paced by `confirm_interval_sec` (`0.8s`) so configured
   `confirm_count` thresholds preserve wall-clock confirmation duration under high-frequency
   tick bursts.
4. The Kalman signal gates normal orders and may initiate configured primary transitions.
   Missing/stale Kalman data preserves the existing fail-open business policy.
5. `_fire_order` validates side and price, resolves the symbol's venue via
   `resolve_symbol_venue`, and submits only through `MarketApi.place` and the common
   `Instrument.place` guard pipeline.

## Retry and submission semantics

Tradeall preserves the historical common persistent retry policy. A refused placement may
enter the generic outbox, while the in-process trend logic also observes its per-trend retry
interval. The retry worker refreshes price and traverses the common guard pipeline before a
new submission. Changing this ownership to strategy-only retry would change financial
behavior and therefore requires a separate explicit decision.

The per-trend maximum counts accepted submissions for anti-spam purposes. Acceptance is
not called a fill and does not prove executed quantity or profit. Daily cap, profit guard,
quantity reconciliation, exchange filters and the shared rapid-fire cooldown still apply.

## Bounds and lifecycle

- coordinator dictionaries are bounded by the configured symbol set;
- trend state contains fixed-size counters/timestamps;
- Kalman fed-price history is capped in `shadow_signals`;
- decision and shadow logs rotate daily and are covered by logger compression/deletion;
- `stop()` wakes the event loop for deterministic test/process shutdown.

## Remaining non-trivial risk

Trend confirmation and accepted-submission counters are in memory. A process restart can
forget the current trend's decision identity and anti-spam budget. The common exchange
cooldown still reduces immediate duplicates, but complete restart safety requires a
durable decision/intent record plus reconciliation of open, partial and terminal venue
orders. That change belongs with the fleet-wide typed lifecycle work and must not be
implemented as a quick local retry patch.
