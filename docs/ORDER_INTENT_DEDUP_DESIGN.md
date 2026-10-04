# Order intent deduplication design

**Status:** Historical design analysis; semantic intent deduplication is not planned.

This document retains the rejected proposal and its failure analysis for reference.

## Current operational decision

`RETRY_DEDUP=false` is intentional in the versioned configuration. The retry
outbox retains every persisted intent as its own record. Semantic equality is not
inferred and is not a planned migration. The former deduplication
key, `(symbol, side)`, was not idempotency: a newer order from another strategy could
replace the older record's quantity, price, options, and client ID revision while the
older age and attempt counters survived. That could silently lose one financial
intent and leave a hybrid record representing neither producer correctly.

With deduplication disabled, each enqueue receives its own record ID and deterministic
client order ID. Leasing, retry, status observation, and terminal reconciliation operate
on that exact record/revision.
The trade-off is that a producer which emits the same logical signal repeatedly can
create multiple records. Queue bounds, guards, cooldowns, price validation, and
venue client IDs reduce the operational risk, but they do not prove semantic equality.

## Historical target key (not planned)

The rejected proposal would have used an explicit `intent_id`, never
`symbol + side`. The proposed canonical identity would have been derived or
persisted from:

```text
origin + venue/account + symbol + strategy_cycle/campaign + kind/tier + side
```

Examples:

```text
monitortrades:binance:TAOUSDC:cycle-18:hard-tp:SELL
assetguardian:binance:TAOUSDC:window-20260826:tier-2:BUY
spot-dca:kraken:TAOUSD:cycle-4:DCA-1:BUY
```

Retries of one intent reuse the same `intent_id` and `client_order_id`. Independent
strategies, accounts, cycles, tiers, and sides remain distinct even when they target
the same symbol.

## Historical proposed record fields (not a gate)

The rejected proposal would have required every producer to publish:

- `intent_id`, `origin`, `venue/account`, `symbol`, `side`, and `kind`;
- requested quantity, order type, requested/reference price, and budget boundary;
- strategy cycle/campaign/tier and an explicit validity/revalidation policy;
- deterministic `client_order_id` where the venue supports it;
- retry policy for absent, ambiguous, rejected, canceled, and expired states;
- terminal policy defining when fills are applied and when the strategy acknowledges
  completion.

## Historical proposed lifecycle rule

The common monitor reconciles persisted intent, venue order, status, fills, and native
terminal reason. It may retry an absent or ambiguous submit only under the intent's
policy. It must not resubmit an open/partial order. Native `REJECTED` and `EXPIRED`
produce a new deterministic client-ID revision for only the unfilled remainder;
`CANCELED` is terminal and alerted because the common layer cannot distinguish an
intentional cancel from a strategy invalidation. The originating strategy remains
responsible for revalidating strategy-specific signals before a new exposure-increasing
intent is emitted.

Acceptance does not remove the outbox record. Accepted trackers ignore submit TTL and
remain durable across status/API failures until terminal venue truth is observed. This
is intentionally fail-closed: a temporarily unreachable exchange cannot turn a real
order into an apparently absent order and trigger another submit.

Trend deferral is not an attempted submit: it consumes neither attempt count nor TTL.
If a different refusal reason later appears, the normal active TTL begins at that
transition.

## Current conclusion

Semantic deduplication has no active migration gate and is not planned.
`RETRY_DEDUP=false` remains the versioned policy. Response-loss, restart,
partial-fill, cancel/fill-race, repeated-signal, and independent same-symbol tests
remain required for mechanical idempotency and recovery correctness; they are not
steps toward merging financial intents.
