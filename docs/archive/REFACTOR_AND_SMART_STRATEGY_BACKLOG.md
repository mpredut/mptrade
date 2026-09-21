# Refactor and Smart Strategy Backlog

**Document version:** 1.0
**Last updated:** 2026-09-03
**Status:** Remaining work; an item is not implemented merely because it is listed here.

## Purpose

This versioned backlog consolidates safe refactors and smarter trading work
identified while hardening `rtrade`, `assetguardian`, `monitortrades`,
`tradeall`, `trailing_stop`, `tradeall_price_archiver`, and the retry
worker. It is not a production configuration or authorization to change
financial behavior.

The target lifecycle is:

```text
market data -> normalized features -> strategy signal -> persistent intent
  -> risk and execution policy -> provider order -> normalized lifecycle
  -> exposure and P&L reconciliation -> restart recovery and observability
```

## Foundations to preserve

- Common quantity decisions: requested quantity, balance, policy and fee caps,
  final quantity, and refusal reason.
- USDC as the primary quote asset in the current fleet configuration.
- Provider-neutral order-state and market-regime interfaces.
- Concurrent paired rounds, persistence, startup reconciliation, and bounded
  state in `rtrade`.
- Safe placement paths with explicit retry ownership.
- Durable retry records with leases, synchronization, client IDs, and provide
  reconciliation.
- Supervision and recovery for active services and the price archiver.
- Reproducible disaster-recovery configuration, with secrets kept outside Git.

These foundations require fleet-wide consistency checks. Their presence in one
bot does not prove that every bot uses them correctly.

## Priority 0 — Complete the order lifecycle contract

### Typed placement result

Replace ambiguous dictionaries, booleans, and provider-specific values with a
`PlacementResult` that includes:

- intent and client order IDs;
- provider order ID, when known;
- normalized state;
- requested and accepted quantity;
- accepted or average fill price, when known;
- retry owner and retry eligibility;
- refusal/error category and provider diagnostics.

It must distinguish request submission, provider acceptance, an open order, and
a confirmed fill.

### Persistent trading intent

Persist a provider-neutral `TradingIntent` before, or atomically with, initial
submission. Record owner, strategy, symbol, side, type, quantity, price policy,
purpose, pair/round identity, timestamps, and a semantic deduplication key.

Saving only after submission creates a crash window where an accepted order has
no local owner. Prefer a lightweight pre-submit intent followed by a post-submit
acceptance update. Benchmark persistence rather than removing this boundary.

### Semantic idempotency

Deduplication must separate intentional repeated trading from replay. Include
owner, strategy, purpose, symbol, side, and correlation identity in the key.
A time window alone is insufficient for paired rounds and recovery actions.
Enable fleet-wide retry deduplication only after restart, timeout, and
unknown-response tests prove the identity model.

### Remaining lifecycle gaps

- Persist and reconcile pending trailing-stop submissions.
- Normalize rejected, open, partial, triggered, filled, and canceled states.
- Define and test `tradeall` restart semantics.
- Reconcile provider state before new speculative work after restart.

## Priority 1 — Normalize provider boundaries

Use common `NormalizedOrder`, `NormalizedFill`, and typed balance models fo
Binance, Kraken, and Hyperliquid. Balances should expose free, locked, total,
and observation time. Strategies must not parse provider payload shapes o
status strings.

Provider adapters should remain mechanical: authentication, symbols, precision,
endpoint parameters, response translation, rate limits, and provider errors.
Quantity calculation, profit policy, retry ownership, and market interpretation
belong in shared layers.

Inventory every live placement call. Most orders should pass through the common
safe-order policy. Narrow exceptions, such as an already-triggered emergency
hard stop, require explicit names and dedicated risk and idempotency tests.

## Priority 2 — Fleet-wide market regime service

Publish one semantic snapshot containing short-, medium-, and long-horizon
direction and strength, confidence, volatility, liquidity, data age, source
health, source disagreement, local evidence, benchmark context, and an explicit
unknown/degraded state.

BTC, ETH, or a wider basket can provide context when local data is sparse, but
must not silently replace the asset's own trend. Lower confidence when local
and benchmark evidence disagree or fallback data is stale.

Regime output may adjust bounded policy values:

- order size and limit-price distance;
- retry/recheck cadence;
- minimum profit threshold and trailing distance;
- maximum concurrent rounds;
- hard-stop urgency;
- whether exposure recovery may use MARKET.

Every dynamic value needs minimum and maximum bounds, a neutral fallback, an
age limit, and an auditable reason. A classifier must never bypass hard risk
limits. MARKET should depend on exposure urgency, adverse movement, spread,
liquidity, volatility, time unpaired, and confidence—not just trend direction.

## Priority 3 — Lightweight portfolio risk coordination

Strict balance reservation is not required for the high-throughput model, but
independent rounds should see:

- free balances and provider positions;
- pending BUY and SELL notional;
- net exposure by asset and quote currency;
- exposure age and unpaired-leg count;
- daily realized/unrealized P&L and fees;
- global and per-strategy risk budgets.

Orders may still resize to available funds at submission. Zero balance should
cause configurable backoff instead of an eight-second request storm. Balance
competition should be an observable policy result, not an unclassified error.

## Priority 4 — Validate smart strategies before promotion

Evaluate `overlay650t8` and `trail_profit_floor_sl18` separately for Kraken
and Hyperliquid. Compare forward net return, fees, maximum drawdown, completed
cycles, open quantity, decision divergence, freshness, and sample size.

Use leakage-free walk-forward evaluation. At each decision, consume only data
available then. Model fees, spread, slippage, partial fills, rejections, balance
caps, retry delays, and provider precision. Compare with the unchanged strategy
under identical execution assumptions and measure both return and tail risk.

Promotion sequence:

1. Unit and characterization tests.
2. Historical event replay.
3. Shadow mode.
4. Small, bounded canary.
5. Gradual production increase.
6. Automatic rollback criteria.

## Priority 5 — Bound archives and long-running state

Operational caches and in-memory containers must stay bounded, but historical
market data should not be discarded just to keep files small. Move dense history
to partitioned append-only storage, preferably Parquet or compressed JSONL by
date and symbol, with atomic checkpoints.

Add gap detection and backfill metadata. A restarted archiver should resume at
its durable watermark and report unrecoverable gaps.

## Priority 6 — Adversarial verification

Add deterministic fault injection for:

- timeout before and after provider acceptance;
- crash before and after persistence;
- partial fill followed by restart;
- cancel racing with fill;
- duplicate retry leases;
- stale data and clock skew;
- exhausted quote or base balance;
- rate limits and temporary bans;
- orphan and duplicate startup orders;
- delayed websocket events, log rotation, and bounded caches.

The core financial invariant is that retries and restarts must not increase
exposure beyond the original authorized intent.

## Priority 7 — Prove disaster recovery

Perform a clean-machine restore using only Git, documented configuration, and
backups. Verify systemd, timers/cron, VPN, DNS, provider connectivity, cache and
database restoration, secret injection, service ordering, health checks, and
safe startup reconciliation.

Never commit tokens or secrets. Document where they are injected and validate
their presence without printing their values.

## Recommended execution orde

1. Typed placement result and normalized order state.
2. Persistent intent and semantic idempotency.
3. Trailing-stop and `tradeall` restart reconciliation.
4. Provider-boundary inventory and removal of bypasses.
5. Common regime snapshot in observation-only mode.
6. Portfolio telemetry and configurable funds backoff.
7. Leakage-free backtests and extended shadow evaluation.
8. Canary only policies that beat the unchanged baseline after costs.
9. Archive migration and a full disaster-recovery drill.

## Design principle

Keep signal, strategy intent, risk policy, execution policy, provide
translation, and financial reconciliation separate. This enables smarte
strategies without letting a classifier or provider quirk bypass safety,
persistence, or accounting.

## Revision history

| Version | Date | Change |
| --- | --- | --- |
| 1.0 | 2026-09-03 | Initial backlog consolidated from the trading-bot hardening thread. |
