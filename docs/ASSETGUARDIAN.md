# assetguardian — financial policy and limits

`assetguardian.py` evaluates each symbol configured through `AG_SYMBOLS`
(`BTCUSDC,TAOUSDC` in the versioned configuration) independently, every 54 seconds.
All `AG_*` keys are mandatory; a missing, empty or invalid value stops the process at
startup. There are no hidden defaults in the code. It is not a stop-loss: crash protection
belongs to the trailing stop module. The Guardian implements two rare contrarian signals,
computed from each asset's own price rather than from the total portfolio value.

## Per-asset signals

- Growth exit in tranches: at +9%/+12%/+15%/+25%/+35% above its own low (scaled dynamically by
  the symbol's 24h volatility multiplier `0.75x..1.25x` and Pillar 3B sentiment) it attempts a
  SELL on that same asset, with progressive allocations `12%/16%/20%/24%/28%` of the free
  quantity captured at the start of the campaign. At the first tranche the window low and the
  initial quantity are frozen; the thresholds do not move as the rolling window advances. The
  SELL profit guard stays active, because a rise above the low does not automatically prove a
  profit above cost.
- Drawdown buy in tranches: at -7%/-10%/-14% below the asset's own 24h high (scaled dynamically
  by the symbol's 24h volatility multiplier `0.75x..1.25x` and Pillar 3B sentiment) it attempts
  a BUY on the same symbol, with progressive pyramid allocations `20%/30%/50%` of that symbol's
  campaign budget. The initial budget is 99.5% of the free USDC when the campaign begins. A
  submit does not complete the tranche: only a terminal `closed` status with an executed
  quantity marks it complete. That asset's campaign re-arms once the drawdown recovers below
  3%, but never while a BUY intent is pending.

Example: a TAOUSDC fall from 242 to 225 is roughly -7.02% and may emit a BUY TAOUSDC intent.
It can no longer produce a BTC intent merely because TAO fell. Mirrored, a TAO rise past the
threshold can sell only the corresponding TAO tranche, not BTC and not the whole portfolio
at the first threshold.

`AG_SELL_REARM_GROWTH_PCT=5` is measured against the frozen low, not a new rolling one. If
the frozen low is 100, the campaign re-arms when the price returns to at most 105. The SELL
state is then cleared; only a future campaign will read and freeze the current low of the
rolling window. The hysteresis prevents immediate oscillation around the first +15% threshold.

The low and the high come from the shared `Price24` cache, and each row is validated for
symbol, timestamp and a finite positive price. A baseline that is missing, stale or made up
solely of the current sample cannot produce an order.

## Execution and retry

Orders go through `mkt.place`/`Instrument.place`. Both AssetGuardian BUY and SELL bypass
`G_QLIMIT1` (`bypass_quantity_policy=True`), because AssetGuardian enforces its own progressive
tranche budgets (`20% -> 30% -> 50%` on BUY, `12% -> 16% -> 20% -> 24% -> 28%` on SELL). The
shallowest BUY tranche (-7% base) respects `G_PROF` (`respect_guard=True`), while deeper BUY
tranches (-10%, -14% base) also bypass `G_PROF` (`bypass_profit_reference=True`). The SELL
keeps `G_PROF`, real balance and fee caps active. The daily cap, anti-spam, cooldown, balance
reconciliation and Binance mechanics all stay active. Neither BUY nor SELL uses the broad
`bypass_profit_guard=True`.

`Instrument.place` no longer uses any `sleep` or polling for the trend: the shared gate
performs a single instant check and, if it refuses, returns control immediately; the shared
outbox will retry on a later tick. AssetGuardian manages the same semantics separately
through `AG_TREND_DEFER_MAX_SEC=180`, which persists a non-blocking deferral per
symbol/side/tranche. Every 15-30s it re-reads the trend, the price, the signal, the balance
and the orders; it places earlier if the trend turns favourable, or after at most 180s and
only if the signal is still valid. No placement function holds the process in a wait loop.

The Guardian sets `caller_owns_retry=True`: it puts no orders into the outbox. Both BUY and
SELL use the generic `TrackedOrderLifecycle` component, available through
`MarketApi.tracked_order_lifecycle`. The strategy persists the intent with a deterministic
`client_order_id` before the submit; the shared component handles lookup, status, TTL cancel
and a provider-neutral audit. `mkt.place` remains the synchronous policy/mechanics call and
does not wait for the fill.

`NEW` and `PARTIALLY_FILLED` stay pending; a tranche is complete only after a terminal
`closed/FILLED` status with `filled_qty>0`. A partially terminal SELL keeps the executed
quantity, and a partially terminal BUY keeps the real cost and fee; a later cycle may
request only the remainder of the tranche. If the submit response is lost, the intent is
looked up by client ID. An explicit confirmation that the order is missing releases it
without completing the tranche and permits re-evaluation or a retry immediately in the same
cycle. Neither an absence nor a lookup error is interpreted as a fill.

For AssetGuardian the "at least once" policy requested by the operator is explicitly
enabled: both a confirmed absence and an unavailable lookup release the intent for a full
strategy revalidation and resend in the same or the next cycle. Resending the same attempt
reuses the same `client_order_id`, so the exchange can deduplicate; but in the case of a
false-negative external read, the policy accepts the risk of a duplicate request rather than
losing the intent. Even this case does not complete the tranche without a confirmed terminal
fill.

A limit BUY/SELL order of the Guardian's own that stays open past
`AG_ORDER_MAX_AGE_SEC=900` (15 minutes) is re-queried and receives at most one cancellation
request. The `cancel_attempted_at` marker is persisted before the API call. After the cancel
the status is requested again: a terminal partial fill is accounted for, and only the
remainder may be retried in a later cycle. If the cancellation or the status is ambiguous,
the intent stays pending and blocks any replacement. This TTL applies exclusively to the
order identified by the Guardian's `client_order_id`; a SELL opened by another module blocks
the Guardian but is not cancelled by it.

The legacy parameters `cancelorders=True` and `hours=...` still appear at other call sites,
but in the consolidated pipeline they are only metadata for the quantity policy and do not
call the old `cancel_orders_old_or_outlier` function. They are not globally reactivated by
this change, so that ownership of `rtrade`, `tradeall` or `monitortrades` orders is not
quietly altered.

A refusal or a confirmed absence triggers a recomputation of the signal, the price, the
balance and the state, without the generic outbox. The interval is dynamic: 54s normally,
30s when within two percentage points of the next tranche, and 15s while a tranche, an
intent or a trend deferral is active. There is no MARKET and no `force=True` in this module.

To limit contention over the shared USDC cash, the evaluator stops the cycle after the first
accepted order. The order of `AG_SYMBOLS` decides which signal is evaluated first; the next
cycle re-evaluates every symbol from the current state. An accepted SELL or BUY order is not
called a fill and does not complete the tranche without exchange reconciliation.

## Persistent state

The state is version 3 and keeps separate `buy` and `sell` campaigns under each symbol. BUY
retains the peak, the initial cash, the executed cost/fee per tranche, the pending intent
and the terminal orders. SELL retains the frozen trough, the initial quantity, the completed
tranches, the executed quantities, the pending intent and at most 20 terminal orders for
auditing. The per-symbol v2 state and the legacy global state are migrated conservatively as
BUY state only; the global legacy belongs solely to the historical BTCUSDC symbol.

## Risks and verdict

- Buying into a drawdown is mean reversion / catching the dip, not protection. It can buy an
  asset that keeps falling.
- The campaigns are separate, but the cash is shared. The first accepted signal reduces the
  balance available to the other assets; each order re-reads the real balance and is capped by it.
- The configuration allows up to a cumulative 99.5% of the available cash for one asset's
  campaign. The tranches reduce timing risk, not concentration risk.
- The shared guards — apart from the BUY reference and the SELL quantity policy, exempted
  narrowly and explicitly — can refuse the order; that is fail-closed behaviour.
- The strategy's profitability is not demonstrated. The change fixes the attribution of the
  signal and the order to the same asset; it does not guarantee a gain.

## Operational invariants

- every `AG_*` key must exist and be valid; there are no fallbacks;
- each decision is `symbol price -> order on the same symbol`;
- a missing baseline or an unavailable balance/price means no order;
- cache rows that are invalid, in the future, stale, `NaN` or infinite are ignored;
- the evaluation reads an atomic snapshot of the cache, without racing the sync thread;
- at most one accepted order per cycle, and no order in the global outbox;
- the same tranche is never accounted twice; its submit may be repeated with the same ID
  after an absent or unverifiable result;
- any pending BUY/SELL intent is reconciled before a new signal is read, even if the price
  feed is momentarily unavailable;
- only a pending order owned by the Guardian may receive a single cancellation once the TTL
  is exceeded; an ambiguous result means pending, with no resubmit;
- an existing open SELL order blocks the creation of a duplicate campaign or order;
- a BUY/SELL trigger reports an accepted submit separately from the fill confirmation;
- quantities are reconciled in the shared pipeline before Binance;
- the logs are handled by the fleet's shared rotation.
