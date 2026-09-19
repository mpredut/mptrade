# rtrade — financial policy, execution and operations

A reference document for `rtrade.py`, `strategies/rtrade_pair.py` and
`rtrade_pair_store.py`. The effective configuration remains `config.env`; the values
below describe the live profile as of 23 August 2026.

## Financial verdict

rtrade is a spot market-making/spread bot on `TAOUSDC`. The mechanics aim to capture the
difference between a BUY below the median price and a SELL above it, while reducing the
operational risk of unpaired orders. **There is, however, no evidence that the strategy has
a positive return or that it maximises profit.**

The exploratory backtest on the TAO cache was negative for every combination of spread and
stop that was tested. The data had a median step of roughly 19 seconds and could not
validate the 8-second fast-fill classification. Enabling it live must therefore be treated
as a forward observation with real capital and risk limits, not as a demonstrated edge.

The mechanisms below serve two distinct purposes:

- profit-seeking: the spread, the profit guard and the fill-anchored exit;
- loss/risk control: per-round ownership, fill reconciliation, the concurrency cap,
  backoff and the emergency exit.

A risk control can reduce accidental losses, but on its own it does not turn a strategy
without an edge into a profitable one.

## Entry and profit target

- The coordinator is active, with at most 4 concurrent rounds.
- A new round may start after a minimum of 8 seconds.
- The directions alternate `BUY-first` and `SELL-first`.
- The requested notional is 500 USDC per round; the final quantity may be smaller.
- The current adjustment is 0.64% on each side of the mid. In theory, before rounding,
  fees and slippage, the BUY-SELL distance is about 1.28% of the mid.
- The minimum exit margin is 1.15%, and the Binance fee cap uses an estimate of 0.1% per
  order. These thresholds protect the requested price, but fill probability, adverse
  selection and slippage can wipe out the theoretical gain.

SELL-first on Binance Spot does not open a borrowed short. It sells only TAO available in
the account and then attempts a buyback. The round's ledger shows a `SOLD` exposure, but at
account level this means the TAO inventory was reduced. The financial risk is buying back
more expensively in a rising market.

## The cycle of one round

1. Compute a BUY and a SELL around the mid.
2. Place both orders through `mkt.place` -> `Instrument.place` -> the Binance provider.
3. Attach a `pair_id` and a deterministic `RT_...` client order ID to each leg.
4. If the second leg cannot be placed, the first is cancelled.
5. With no fill by the TTL (32 seconds), both are cancelled and the state is re-read to
   close the fill-versus-cancel race.
6. With a single fill or a partial fill, the entry remainder is cancelled, the net exposure
   is recomputed from the exchange's fills, and the opposite exit is resized to that quantity.
7. Other rounds may start meanwhile; each owns its orders and its ledger separately.
8. A round becomes terminal only on a complete pair, expiry without a fill, a controlled
   failure, or an executed hard stop.

Rounds reserve no balance between them. The pipeline recomputes the balance before each
submit, so concurrency produces a controlled clamp or refusal, but it can reduce the
probability that both legs of the same round are accepted.

## Guards and quantity

Normal limit orders go through the shared pipeline:

```text
requested_qty
  -> balance_cap
  -> daily/weight policy_cap
  -> fee_cap
  -> Binance precision and minimum
  -> final_qty or refuse_reason
```

The profit guard, the daily cap, trend-wait, the per-`pair_id` cooldown, the free balance,
the fee cap and the Binance validations all apply. On a zero balance or another refusal,
the direction enters a 180-second backoff. `caller_owns_retry=True` excludes rtrade orders
from the global outbox; the coordinator is the sole owner of retry and reconciliation.

These guards are conservative. They can avoid disadvantageous trades, but they can also
greatly reduce the number of fills. For instance, reaching the daily cap can leave rtrade
healthy but with no active orders.

## Exposure, trend and stop

After a one-sided fill, the limit exit is anchored to the real average entry price and to
the minimum edge; it does not automatically chase the market into a loss.

The current thresholds are:

- fast-fill/shock: a fill within at most 25% of the TTL, that is roughly 8 seconds;
- shock hard-stop evaluation threshold: 4%;
- normal hard-stop evaluation threshold: 8%;
- emergency threshold: 12%.

Under `RTRADE_DYNAMIC_MARKET_EXIT_MODE=live`, 4% and 8% do **not** guarantee a MARKET order.
At those thresholds MARKET is permitted only if `MarketRegimeDecision` confirms a trend
adverse to the exposure. If it does not, the round keeps the anchored exit and waits. At
12%, the emergency permits MARKET regardless of the signal. This reduces panic selling into
a temporary move, but it explicitly accepts tail risk between the initial threshold and 12%
if the trend detector is wrong or late.

MARKET is permitted only to reduce an exposure that already exists. The quantity is
reconciled once more against the balance, the fee cap, the precision and the venue minimum.

## Persistence and recovery

`cachedb/rtrade_pairs.json` keeps the canonical intent before the submit, the requested and
accepted values, the order ID and the coordinator checkpoint. LIMIT orders go through
`order_retry.TrackedOrderLifecycle`, but they stay in the rtrade state rather than the
global outbox. Each lifecycle call performs a single submit and does not wait for a
terminal status. If the response carries no order ID, rtrade performs an immediate lookup;
only a confirmed absence permits a second call with the same client ID. There is no submit
loop and no busy waiting. On restart:

- intent plus an existing order: the order is adopted;
- an intent without an order after a confirmed absence: an idempotent submit with the same client ID;
- a lost submit response: a lookup by client ID, with no concurrent submit;
- an `RT_` order without local ownership: automatic cancellation and confirmation;
- an ambiguous state or an unavailable API: fail-closed, with no speculative order.

At most 200 terminal rounds are kept. Active checkpoints are written in batches, terminal
orders with zero fills are compacted, and the auxiliary caches have memory caps.

The shared lifecycle solves the mechanics of persistence, lookup and status.
`PairCoordinator` remains the owner of the TTL, cancel/reprice, partial fill and hard-stop
policy. The legacy `repetitive_buy`/`repetitive_sell` path stays available behind the
feature flag and is not consumed by the global worker.

## Operational invariants

- at most 4 active coordinators and a minimum of 8 seconds between rounds;
- exactly one owner (`pair_id`) for each rtrade leg;
- no MARKET exit outside an existing exposure and the stop policy;
- P&L and `net_qty` computed from fills, not from the requested quantity;
- the same intent produces the same client order ID after a restart;
- rtrade never uses the local retry and the global outbox at the same time;
- a lack of certainty about the exchange blocks a new submit rather than inventing state.

## What to watch for financial validation

- net P&L after fees and slippage, not gross cashflow;
- the rate of fully paired rounds and the average time spent one-sided;
- the loss conditional on BUY-first versus SELL-first;
- fill probability at the 0.64% spread;
- how many entries are refused by the profit guard or the daily cap;
- drawdown and the loss at the extreme percentiles;
- the trend decisions at the 4%/8% thresholds and the emergency executions at 12%;
- enough forward results before increasing the notional or the concurrency.

The deterministic tests validate the mechanics and the recovery, not profitability. Any
change to the spread, the stops, the notional or the number of rounds requires a replay,
the full test suite and forward observation.
