# Centralising OrderLifecycle

**Architecture decision updated:** 2026-10-04

This document centralizes order mechanics only. A global/common financial
ledger, a global/common portfolio-risk coordinator, and semantic intent
deduplication are not planned. Exact-record idempotency and strategy-owned
financial state remain required.

## The current decision

`order_retry.py` is the canonical source for the reusable mechanics of the order
lifecycle. The centralisation concerns execution mechanics, not the strategies'
financial decisions.

There are two ways of using the same domain:

1. the global outbox: `Instrument.place` persists before the submit, and
   `order_retry_worker.py` is the single consumer that resumes and tracks the records;
2. a lifecycle with state owned by the strategy: the strategy synchronously calls
   `order_retry.TrackedOrderLifecycle.submit/reconcile`, supplying the callback through
   which the intent is persisted atomically into its campaign.

`TrackedOrderLifecycle` is not, and does not start, a separate process. It is a state
machine called once per tick by AssetGuardian, spot-DCA and trailing. The fleet's
continuous process remains `order_retry_worker.py`, and it consumes only the global outbox.
The worker is a separate OS process, not a thread started by `Instrument.place`.
Placement does no polling and does not wait for the terminal status: it finishes after
persistence, the guards and a single submit call to the provider.

The former `providers/tracked_order.py` compatibility shim had no active consumers and
contained only re-exports, so it was removed. Production code imports lifecycle types
directly from `order_retry`.

## What is centralised

- the typed submit outcome: accepted, definitively refused, or unknown;
- validating the `intent_id` / `client_order_id` identity;
- persistence before the submit;
- the distinction between acceptance and fill;
- recovering from an ambiguous response through the client order ID;
- the `open`, partial and terminal statuses;
- the outbox's persistent transitions from a status snapshot;
- preserving the venue's native status;
- multiple confirmations of absence;
- the bounded cancel, persisted before the external effect;
- the mechanical audit of the lifecycle stages;
- the adapter for a credential-scoped `StrategyExecutor`.

## What stays in the strategy

- the signal and its validity;
- the campaign, the cycle, the tier and the budget;
- accounting for fill deltas and P&L;
- applying the fill to the position;
- deciding whether a terminal status permits a new intent;
- the difference between ENTRY, DCA, TP and a protective exit;
- whether a cancel invalidates the intent or represents repricing;
- whether an exit may move from LIMIT to MARKET.

A strategy uses `caller_owns_retry=True` when it keeps its intent in its own state and
calls the central lifecycle. The flag does not mean the strategy reinvents the mechanics;
it means the global outbox may not later create an intent after the strategy has
invalidated its campaign.

## Inventory after the code centralisation

| Path | Lifecycle engine | Active persistence |
|---|---|---|
| `tradeall`, `monitortrades`, commands through `Instrument.place` | outbox plus the worker from `order_retry` | `cachedb/order_retry_queue.jsonl` |
| AssetGuardian BUY/SELL per asset | `order_retry.TrackedOrderLifecycle` | the AssetGuardian campaign |
| Binance/Kraken trailing | `order_retry.TrackedOrderLifecycle` | the trailing state |
| Kraken/Hyperliquid spot-DCA | `order_retry.TrackedOrderLifecycle` plus DCA accounting | the strategy's state |
| rtrade pair LIMIT | `order_retry.TrackedOrderLifecycle` for the submit and startup recovery; the coordinator keeps the TTL/cancel/fill policy | `cachedb/rtrade_pairs.json` |
| rtrade hard stop | audited execution and recovery through the lifecycle; the MARKET policy stays in the coordinator | `cachedb/rtrade_pairs.json` |
| T212 strategy | its own lifecycle based on active orders and the position delta | the T212 state |
| T212 one-shot | its own reconciliation, without a universal client ID | a profile plus ticker marker |
| `monitororder` | not live; no longer extended | excluded from the next migration |

## Declared reconciliation capabilities

The adapters can no longer let the lifecycle infer support from the accidental existence of
a method or from an inherited `[]`. `OrderReconciliationCapabilities` strictly declares the
normalised operations available; a missing declaration means no support.

| Venue | client ID lookup | order ID status | order ID cancel | open orders list |
|---|---:|---:|---:|---:|
| Binance | yes | yes | yes | yes |
| Kraken | yes | yes | yes | yes |
| Hyperliquid spot | yes | yes | yes | yes |
| Trading212 | no | yes | yes | no, reconciliation stays order plus portfolio |

`false` does not mean the venue has no possible endpoint; it means the shared adapter does
not currently offer an operation strict enough for the lifecycle. Transport errors stay
distinct from a missing capability.

## Why financial state remains distributed

A global/common financial ledger is not planned. Moving state authority out of a
strategy could resend a BUY after its signal expired, lose a protective exit after
an external cancel, or apply a fill to the wrong campaign. The shared layer owns
mechanical transitions; each strategy remains authoritative for campaigns, cycles,
positions, budgets, and terminal financial policy.

This boundary is an architectural decision, not an intermediate migration stage.
Characterization tests protect it when mechanical code is shared.

T212 additionally requires different recovery capabilities: active orders and the
portfolio delta, because client order ID lookup is not universally available.

`active_intents.py` is a read-only normalized index over existing strategy-owned
files, not the first stage of a common ledger. It performs no writes and has no
submit, cancel, retry, repair, or policy authority. Missing or malformed sources
are reported as read errors. Strategy files remain the only financial source of
truth. Its aggregate notional and exposure summary is operational observability,
not a global portfolio-risk policy or enforcement engine.

T212 now reuses the common typed and audited submit boundary. Its venue-specific recovery
is intentionally retained: one unique matching active order proves acceptance, while a
portfolio delta independently proves execution. Absence is confirmed across snapshots before
the strategy is allowed to re-evaluate the same financial decision.

## Mechanical idempotency is not semantic deduplication

An `intent_id`, record ID, deterministic `client_order_id`, and venue order ID
identify and reconcile one exact submission. They do not authorize merging two
strategy decisions that happen to share a symbol and side. The versioned retry
configuration keeps `RETRY_DEDUP=false`; separate intents remain separate records.

## Typed submit outcomes do not change retry policy

The common boundary records `accepted` only when a venue order ID exists, `refused` only
when the adapter can prove synchronous non-acceptance, and `unknown` for transport errors or
responses without an order ID. Existing retry cadence, TTL, price gates, attempt limits and
terminal-state decisions are unchanged. The richer state prevents an unknown submit from
being mislabeled as a refusal; recovery still queries venue truth before another submit.

## Optional strategy-local terminal policy — deferred

A strategy may eventually use a declarative per-intent policy inside its own
durable state. This would not create a global ledger, global portfolio policy, or
semantic deduplication layer. The candidate fields retained for local analysis are:

```text
intent_id
origin / strategy
venue / account
symbol / side / kind
cycle / campaign / tier
requested_qty / executed_qty / remaining_qty
requested_price / budget_cap
valid_until / state_version
retry_on_absent
retry_on_rejected
retry_on_expired
retry_on_canceled
partial_fill_policy
cancel_origin
requires_strategy_revalidation
protective_exit / allow_market_fallback
```

The status alone does not decide the retry. `CANCELED` can mean a strategy invalidation,
repricing, the operator, the exchange, or an unknown state. That is why the financial
policies will not be added as a generic fallback in `order_retry`.

## The remaining refactor steps

1. preserve characterization and golden tests for every venue-specific recovery path;
2. extract only duplicated mechanical transitions whose semantics are demonstrably equal;
3. keep financial policy and durable authority with the owning strategy;
4. do not introduce a global ledger, global portfolio-risk engine, or semantic deduplication.

The duplicated mechanical transitions in `order_retry_worker.py` were extracted into
`order_retry.advance_claimed_status`. The worker keeps only the venue I/O, the audit and the
orchestration of one iteration; the shared core applies the snapshot to the outbox atomically.

No thresholds, budgets, tiers or financial semantics change during this mechanical
centralisation.
