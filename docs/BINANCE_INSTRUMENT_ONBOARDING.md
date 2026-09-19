# Binance instrument onboarding and local handoff

Updated: 2026-09-07. Implementation and tests are local; production is not verified.

## Single configuration entry

Use `instruments.conf` for identity, enabled membership, consumer opt-ins, and
per-coin policy. `instrument_registry.py` is credential-free and provider-independent;
`instruments_config.py` binds selected records to the existing provider facade.

An enabled Binance instrument automatically joins public price subscriptions,
account caches, TradeAll trend tracking, and the default observer/report symbol list.
That is **not** permission for TradeAll, AssetGuardian, or trailing to place orders.

| Role | Consumer / meaning |
| --- | --- |
| `role.tradeall_fire` | Allows attempts through TradeAll's existing guarded `_fire_order` path |
| `role.kalman_primary` | Allows Kalman transitions to initiate TradeAll attempts; requires `tradeall_fire=yes` |
| `role.trailing` | Binance trailing adapter; requires an explicit `trailing.pct` |
| `role.assetguardian` | AssetGuardian evaluates its existing per-asset campaign policy |
| `role.mt` | `monitortrades`; configure its existing `mt.*` policy in the same section |
| `role.archive` | Default long-price archive membership |
| `role.rtrade` | The single pair owned by rtrade; its startup requires exactly one selected pair |
| `role.force_sell` | Membership of the existing force-sell eligibility list |

Every role flag is required, including explicit `no` values. Missing settings,
invalid booleans, unknown roles, duplicate provider/symbol identities, invalid
trailing percentages, or missing/invalid TradeAll Kalman modes raise errors.
The current Binance fleet is USDC-based; adding a new quote currency requires a
separate sizing/cash-policy review, not merely changing `quote`.

Example for **observation only**, using a fictitious symbol (replace with a real
exchange-supported USDC pair before enabling):

```ini
[BINANCE_NEW]
provider = binance
symbol = NEWUSDC
base = NEW
quote = USDC
enabled = yes
isolation = own_ledger
market_hours = 24x7
role.mt = no
role.tradeall_fire = no
role.kalman_primary = no
role.trailing = no
role.assetguardian = no
role.archive = no
role.rtrade = no
role.force_sell = no
```

Set only the intended roles to `yes`. Add `trailing.pct` if enabling trailing,
`tradeall.kalman_mode = strict|permissive|off` if enabling TradeAll, and the
required `mt.*` parameters if enabling monitortrades. Thresholds and budgets are
financial choices, not guessed defaults. All role flags set to `no` is valid.
`enabled=no` excludes a record from active consumers.

Do not edit `symbols.py`, the adapters, or environment symbol lists when adding
a Binance coin. Historical BTC/TAO/HYPE aliases remain compatibility labels only.
`AG_SYMBOLS`, `TRADEALL_FIRE_SYMBOLS`, `KALMAN_PRIMARY_SYMBOLS`, and
`KALMAN_GATE_MODE*` environment overrides no longer configure membership/policy;
move any intentional server overrides into the registry before deployment.
Explicit `--symbols` CLI overrides for observation/archive jobs remain supported.

This change centralizes **Binance onboarding**. The registry already selects
cross-venue monitortrades records, but standalone Kraken/Hyperliquid/T212 strategy
configurations are not migrated into this file by this batch.

Legacy implicit venue routing still reserves HYPE-prefixed symbols for
Hyperliquid. A Binance symbol overlapping that alias family needs explicit venue
routing work before activation; it is not covered by the generic USDC example.
The current BTC/TAO/ARB selections do not overlap. This consolidation does not
silently reassign an existing cross-venue symbol to a different exchange.

## Preserved current selections

- BTC: TradeAll strict mode, primary Kalman, trailing 20%, AssetGuardian,
  monitortrades, long archive, and force-sell eligibility.
- TAO: TradeAll permissive mode, trailing 22%, AssetGuardian, monitortrades,
  long archive, rtrade, and force-sell eligibility.
- ARB: price/trend observation and trailing 13% only. No TradeAll, AssetGuardian,
  monitortrades, rtrade, force-sell, or long-archive opt-in.
- Existing Kraken/Hyperliquid/T212 enabled states and monitortrades financial
  parameters remain unchanged.

No peak, warm-up threshold, rebuy, pending order, cache, or retry state is migrated
or cleared. The original execution gates, quantities, budgets, and Kalman
missing/stale-signal behavior are unchanged. In particular, unknown trend means
optional trailing filters do not block; it is not proof of financial safety.

## Local verification and production resumption

Offline checks, using the checkout's Python environment:

```bash
.venv/bin/python instrument_registry.py
bash deploy_providers.sh --check
.venv/bin/python -m pytest -q tests
```

Tests must use empty/inert exchange credentials and disabled automatic WebSockets;
the test harness disables external notifications and isolates execution audits/retry
queues. The onboarding regression adds a temporary coin section and imports the
real consumers without API clients or orders.

When server access returns, first inspect its Git status, local configuration
overrides and state. Do not overwrite server edits. Compare the running revision
with the intended commit and verify the correct checkout/venv. A push alone is
not deployment.

A restart of `binance.service` refreshes only `role=fleet`. Binance trailing is
`role=bot`; it must also restart. The updated `deploy_providers.sh` refreshes
both roles, stops on pull/preflight failure, and requires replacement PIDs and
fresh caches on consecutive checks. `restart_bots.sh` coordinates with the
healthcheck supervisor, and shared process helpers restrict matching to the
current user and manifest working directory. A process that cannot stop
gracefully causes failure instead of a forced kill or duplicate launch.
This workflow assumes the existing fleet supervisor is active.

After deployment, inspect:

1. Fresh account-cache health, price/trend timestamps, retry queue, and new log errors.
2. `binance_api/trailing_stop.py --status`: persisted ARB peak, `warmup_at`,
   rebuy, and pending order. Status does not prove process health or live prices.
3. ARB is actually tracked by the newly started Binance trailing process, with
   valid balances/quotes and no conflicting order owner.
4. The REST price accessor uses bounded-age cached quotes and refreshes over REST;
   it does not necessarily make a new network call on every tick.

If runtime state is absent, warm-up uses the provider's inventory-reconciled
acquisition cost when available. Binance reads fresh, version-matched immutable
BUY/SELL fills, includes base/quote commission effects, and requires the resulting
quantity to match free + locked holdings. Fully sold cycles no longer pollute the
new position's average. Unsupported providers, inconsistent history, or stale
caches retain the first-observed-price fallback. Non-finite cost references are
rejected. Fees paid in third-party assets are not converted into quote cost;
matching quantities does not prove a complete external transfer history.

Existing saved warm-up, peak, rebuy, and pending-order state is not recalculated.
Do not assert that ARB is already armed or automatically pre-seed a historical peak.

## Consolidated upstream work

The local refactor is rebased on upstream through `41a34ed`, preserving the
`232ffc4`, `4490754`, `3de6737`, and `41a34ed` history. The upstream
`binance_symbols`, `trail_pct_map`, and `tradeall_trade_symbols` APIs remain
thin compatibility wrappers over the one registry. Obsolete keys `trail.enabled`,
`trail.pct`, and `tradeall.trade` must be migrated to `role.trailing`,
`trailing.pct`, and `role.tradeall_fire`; they are rejected if left behind,
not silently used as a second source. The committed registry is already migrated.

The upstream bot-launcher pipe fix is preserved: deployment redirects the launcher
to `logs/deploy_restart_bots.log` and checks its exit status before verification.
Financial-floor review: [Profit-floor assessment](PROFIT_FLOOR_REVIEW_2026-09-07.md).

## Additional findings to review separately

The legacy `verify_tools/pnl_report.py` now includes registry Binance symbols,
but its existing cross-venue valuation still contains a hard-coded Kraken HYPE
price and its historical Binance P&L excludes fees/older cost basis. Do not treat
its aggregate output as an authoritative live valuation. Correct valuation is a
separate review; this batch changes only its Binance symbol membership.
