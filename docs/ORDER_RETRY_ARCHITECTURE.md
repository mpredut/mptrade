# Order retry, lifecycle and strategy state

This is the canonical design for persistent orders. `order_retry` centralises the order
mechanics; the strategy remains the authority for the signal, the budget, the campaign and
the financial decision that follows a terminal status.

## The two flows

### 1. The global outbox

```text
a strategy without its own lifecycle
  -> Instrument.place
  -> persists the intent into cachedb/order_retry_queue.jsonl
  -> performs a single submit, with no terminal polling
  -> order_retry_worker.py reads the outbox later
  -> lookup/status/retry according to the record's state
```

`order_retry.py` is a library. It defines the outbox format, the atomic operations and the
shared state machine; it does not run on its own. `order_retry_worker.py` is the single OS
process that consumes the global outbox. Only its main loop calls `process_once`; an
`Instrument.place` call neither starts the worker nor waits for the order to become terminal.

`RETRY_DEDUP=false` keeps a separate record for each intent. Two independent intents on the
same symbol and the same side do not overwrite each other. The idempotency of a single
intent comes from `intent_id` and `client_order_id`, not from deduplication by
`symbol + side`.

### 2. A lifecycle owned by the strategy

```text
a stateful strategy
  -> TrackedOrderLifecycle.submit
       -> an atomic callback into the strategy's JSON
       -> a single submit
  -> the next tick or restart
       -> TrackedOrderLifecycle.reconcile
       -> client ID lookup, status and terminal state
  -> the strategy applies the fill and updates the campaign
```

This flow uses `caller_owns_retry=True`, so the order does not also enter the global outbox.
There are never two concurrent monitors. `TrackedOrderLifecycle` is a class called during the
ticks of the strategy's process, not a thread and not a hidden daemon.

## What the cache threads in the worker process are

An `order_retry_worker.py` process can end up with several threads once it touches the
Binance path. They are not additional consumers of the queue:

- `MainThread` owns the single `process_once` loop and reads the JSONL;
- `BinanceTimeResync` maintains the clock offset for signed requests;
- the managers created lazily by `cacheManager.CacheFactory` update caches such as orders,
  fills/trades or the current price, when the profit guard and the Binance pipeline ask for them;
- the trend/cache threads (`InstantTrend...`, price managers or the websocket) appear only if
  that infrastructure is initialised in the process.

Each `CacheManagerInterface.periodic_sync` has its own daemon thread. It feeds the data used
by the guards; it does not read `order_retry_queue.jsonl`, does not resend orders and does
not change lifecycle ownership. The exact thread count depends on which caches are requested
lazily in that process. Separating a minimal Binance provider for the worker could reduce
these threads, but it must be characterised first, because the current guards depend on the
history held in the cache.

## The shared contract of an intent

The important mechanical fields are:

```text
intent_id, client_order_id, venue, symbol, side, kind
requested_qty, requested_price, attempt, created_at
order_id, submitted_qty, submitted_price, submit_status
lookup_misses, last_status, filled_qty, terminal_status
```

The mandatory order is: durable persistence, then the external effect. A lost submit response
leaves the intent persisted without an `order_id`; reconciliation looks it up by
`client_order_id` first. A lookup error is ambiguous and blocks resending. Only an absence
confirmed according to the owner's policy may release the intent for a retry.

## rtrade after the refactor

The active `PairCoordinator` path now uses the shared lifecycle for LIMIT orders and for
startup recovery:

```text
PairCoordinator
  -> _LivePairVenue.place_limit
  -> TrackedOrderLifecycle.submit
  -> RTradePairStore.persist_intent
  -> mkt.place(... caller_owns_retry=True)
```

The submit does no polling and does not block until the fill. If the first response carries no
`order_id`, the live path performs a single immediate reconciliation, with no sleep loop. A
found order is adopted; only after a confirmed absence is a second idempotent submit with the
same client ID permitted. An unavailable lookup or status keeps the intent and blocks new
rounds. On restart, `recover_intent` also converts the old records into the canonical format,
looks the order up by client ID, and reads the normalised status.

`PairCoordinator` deliberately keeps the per-tick policy for the TTL, cancel, repricing,
partial fills and the hard stop. Those are the round's financial decisions, not generic
mechanics. `repetitive_buy` and `repetitive_sell` remain the legacy path behind the feature
flag and have not been removed.

## Where the state is saved

The files are relative to the repository root, unless a path is set through configuration:

| Owner | File/pattern |
|---|---|
| the global outbox | `cachedb/order_retry_queue.jsonl` plus `.lock` |
| rtrade | `cachedb/rtrade_pairs.json` plus `.lock` |
| AssetGuardian | `statedb/bots/assetguardian_state.json` |
| Binance trailing | `statedb/bots/trailing_state_binance.json` |
| Kraken trailing | `statedb/bots/trailing_state_kraken.json` |
| Hyperliquid trailing | `statedb/bots/trailing_state_hyperliquid.json` |
| Kraken spot-DCA | `kraken/.state_<PAIR>.json` |
| Hyperliquid spot-DCA live | `hyperliquid/.state_<TOKEN>.json` |
| Hyperliquid spot-DCA paper | `hyperliquid/.paper_state/.state_<TOKEN>.json` |
| Hyperliquid legacy directional | `hyperliquid/.state_<COIN>_<direction>.json` |
| Hyperliquid delta-neutral | `hyperliquid/.state_dn_<COIN>.json` |
| Trading212 per ticker | `212trading/.state_<TICKER>.json` |

For rtrade, the JSON has `pairs[pair_id]`. Each round holds the identity, the quantity, the
phase, `intents`, the `state` checkpoint, the timings and the `terminal` marker.
`intents["limit:BUY"]`/`intents["limit:SELL"]` hold the canonical lifecycle;
`state.tickets` and `state.snapshots` hold the coordinator's financial ledger.
Writing uses a file lock, a temporary file, `fsync` and `os.replace`.

## What reconciliation means

Reconciliation does not mean "assume the order succeeded", nor is it a simple balance
comparison. It is the deterministic correlation of four sources:

1. the persisted intent (`intent_id` and `client_order_id`);
2. the venue's order (`order_id`, open orders and lookup by client ID);
3. the normalised status and the cumulative fills (`filled_qty`, cost, fee);
4. the strategy's financial state and the execution audit.

If the `order_id` is missing, a lookup by `client_order_id` follows. If the order exists, the
ID is persisted and the status is read. If it is open or partial, it stays tracked without a
resubmit. If it is terminal, the venue's truth stays in `terminal_status` until the strategy
atomically applies the fill delta to the position and the checkpoint. If the lookup or the
status fails, the state is ambiguous and is preserved; absence is never invented.
The open-order inventory separately detects orphaned orders that have no local owner.

This boundary avoids both the lost order and uncontrolled resending, without moving the
thresholds or the financial policy into a generic fallback.

## A local refusal before the submit versus an ambiguous result after it

Not every `Instrument.place(...)` call that returns no `order_id` means the order was lost.
There are two different situations:

1. **A local refusal before the submit.** The profit guard, the trend, validation or another
   policy can stop the intent before the provider is ever called. The lifecycle state is
   `submit_refused`, with the exact reason in `outcome_context`. There is no order on the
   exchange to be looked up or resent by the worker.
2. **A lost or ambiguous response after the submit.** The provider may have received the
   order, but the process did not receive a reliable `order_id`. Here a lookup by
   `client_order_id` comes first; resending is permitted only if reconciliation does not find
   the existing order and the intent's policy allows a retry.

For `rtrade`, local refusals stay in the audit as terminal intents and are not put into the
generic outbox. The current pair is closed, and a later evaluation may create a new intent
after the backoff. This stops the generic worker from trying to reconstruct the relationship
between the BUY and the SELL outside the strategy.

## The rtrade identifiers

An `rtrade` pair uses four identifiers with different roles:

| Field | Example | Role |
| --- | --- | --- |
| `pair_id` | `3110e4e9a39f4453b0acaf1d41c9537a` | The persistent internal identity of the BUY/SELL pair. It is the primary key in `cachedb/rtrade_pairs.json`. |
| `intent_id` | `rtrade:3110...:limit:buy` | The internal correlation key between the strategy, the audit and the lifecycle. It is not sent to the exchange and is not bound by the Binance limit on `newClientOrderId`. |
| `client_order_id` | `RT_f3d1...` | The idempotency key we choose and send to the provider. The `RT_` prefix identifies the owner, and the rest deterministically identifies a single `pair_id + side + kind` branch. |
| `order_id` | a number allocated by the venue | The identity returned by the exchange once the order is accepted. |

In the current implementation, `client_order_id` is:

```text
RT_ + BLAKE2s-128(pair_id + side + kind)
```

The result is 35 characters and fits within the 36-character limit handled by the Binance
integration. The 128-bit hash is not needed for secrecy; it is used as a deterministic,
compact key with a negligible collision probability. The same branch produces the same ID
after a restart, and BUY and SELL produce different IDs without a global counter.

`RT` on its own cannot be a `client_order_id`: every rtrade order would share the same key,
lookups would become ambiguous, and the provider may refuse to reuse it. The form `RT_123`
would be safe only if `123` came from a persistent, atomic allocator, unique across processes
and restarts, with the value saved before the first submit. A counter kept only in memory
would reset after a reboot and could misidentify an old order.

For operational readability, the recommended solution is to keep the protocol ID stable and
add a separate display alias, for example `R000123`, or to show the first 8 characters of
`pair_id`. That alias may appear in logs and reports, but it must not be used on its own for
idempotency or reconciliation. The same rule applies to `intent_id`: the long form stays the
exact key, and a short alias is for the operator only.

Identifiers that have already been persisted are immutable. A schema change must be versioned
and migrated only when no orders or intents are in flight; otherwise a lookup by the old
`client_order_id` can miss precisely the order that reconciliation is trying to protect.

## rtrade reconciliation after a restart

At startup, `rtrade` reads `cachedb/rtrade_pairs.json` and treats every non-terminal record as
a state to recover, not as a reason for a new submit. For each branch it correlates:

1. `pair_id` and `intent_id` from the strategy state;
2. `client_order_id` from the lifecycle/audit;
3. `order_id` and the normalised status from the provider cache or the live lookup;
4. the requested, executed and remaining quantity, including partial fills and fees;
5. the branch's terminal state and its effect on the BUY/SELL pair.

Only after that correlation is one of the terminal actions chosen: track the existing order,
apply the fill into the persistent strategy, resend the same intent according to its policy,
or close the refused/cancelled intent. A reboot must never create a new `pair_id`, `intent_id`
or `client_order_id` for a branch that was already issued.
