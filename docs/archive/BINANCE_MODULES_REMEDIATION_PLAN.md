# Binance Modules Remediation Plan

Internal audit artifact. This document is a plan only; it does not authorize code,
configuration, process, deployment, or live-order changes.

Status snapshot (2026-08-23): B07's immediate correctness fixes are deployed on
`main@42ee397`: guarded refusal handling, unavailable-balance semantics, the global
recent-trade gate, finite input checks, post-trade BUY cap, own-ledger SELL cap and
caller-owned strategy retry. Durable hard-TP lifecycle/cooldown reconciliation remains
open and belongs with B01/B02; it is not a quick follow-up.

## Audit basis and invariants

- Scope: active Binance fleet, Binance API adapter, shared execution and persistence
  layers used by Binance, supervision, recovery, and tests. Cross-venue modules are
  included only where a shared-layer change can affect Binance behavior.
- Preserve the business decision that active rounds do not reserve balances. Concurrent
  rounds may compete for free funds; the final provider preflight is the authority.
- Treat exchange state as authoritative. An accepted order is not a fill, and a missing
  response is an unknown outcome, not a confirmed failure.
- Most placement must traverse `Instrument.place` and the common guarded pipeline.
  Mechanics-only provider calls require an explicit, audited exception.
- Never retry a non-idempotent submission blindly. Recovery first reconciles by client
  order ID, venue order ID, open orders, and fills.
- Quantity, price, fees, and limits must use finite positive values and exchange filters.
- State and resource growth must be bounded. Critical transitions require durable,
  restart-safe persistence.
- Each implementation batch below must pass its tests and a second adversarial review
  before the next batch begins.

## Execution loop

For every batch, in order:

1. Capture characterization tests for current intended behavior.
2. Add failure-injection tests for timeout, malformed data, partial fill, restart, and
   concurrent callers where applicable.
3. Implement one bounded change without mixing unrelated refactors.
4. Run targeted tests, provider contract tests, and the full suite on DEV/backtest.
5. Run static ownership scans for direct order API calls and unbounded structures.
6. Review financial behavior: exposure, fees, fill semantics, and worst-case recovery.
7. Commit the batch separately. Do not deploy until the batch-specific gate passes.
8. For runtime changes, deploy canary-style, reconcile exchange state, then observe
   health, order lifecycle, balances, logs, and restart recovery before proceeding.

## P0 - Financial correctness and external truth

### B01. Typed placement outcome across the common pipeline

Affected: `instrument.py`, `providers/market_api.py`, `providers/strategy_executor.py`,
`providers/execution_audit.py`, Binance provider and all active Binance callers.

Problem:
- Placement currently returns a mixture of order dictionaries and `None`; guard refusal,
  provider rejection, transport uncertainty, accepted order, partial fill, and filled
  order can collapse into indistinguishable caller behavior.
- Several bots mark work complete when an order was merely accepted.

Plan:
- Introduce a single structured outcome with intent ID, client order ID, venue order ID,
  normalized lifecycle state, requested/effective/executed quantities, average price,
  refusal reason, retry ownership, and an `outcome_unknown` state.
- Preserve compatibility only through a short-lived adapter with deprecation tests.
- Make every caller branch on normalized state rather than truthiness.

Gate:
- Contract tests for rejected, unknown, accepted, partial, filled, canceled, expired, and
  duplicate-client-ID results on every provider adapter used by the fleet.

### B02. Make submission recoverable and idempotent

Affected: common strategy executor, execution audit, Binance adapter, `rtrade.py`,
`rtrade_pair_store.py`, `assetguardian.py`, `tradeall.py`, trailing adapters.

Problem:
- A transport timeout after submission can be an accepted order with a lost response.
- The generic execution audit is observational, not a durable recovery journal.
- Some bots have durable intent state; others only retain in-memory state or accepted
  response state.

Plan:
- Persist intent before submit, including owner/module, symbol, side, quantity, policy,
  pair/campaign ID, deterministic client order ID, and transition version.
- After any exception or restart, query by client order ID before deciding to submit,
  cancel, replace, or close the intent.
- Persist transition to submitted/terminal after exchange reconciliation. Use atomic
  replace plus file and directory durability for financially critical records.
- Define owner-specific recovery policies; never let a generic worker reinterpret a
  strategy signal after its validity window.

Gate:
- Kill-point tests before submit, after venue acceptance, before local save, during
  partial fill, and during cancel/replace prove no duplicate order and no lost exposure.

### B03. Close mechanics-only bypasses

Affected: `providers/market_api.py`, `binance_api/bapi_placeorder.py`, verification tools,
all direct `client.create_order`/market/limit helpers.

Problem:
- The public mechanics-only facade can bypass guards if a new caller uses it incorrectly.
- Low-level Binance helpers remain callable and overlap with the common safe path.

Plan:
- Make raw mechanics private or require an explicit internal capability supplied only by
  the common executor.
- Maintain a machine-checked ownership inventory of every direct submit/cancel site.
- Remove dead legacy adapters after call-graph and characterization confirmation.
- Keep explicit emergency exceptions narrow, named, logged, and independently tested.

Gate:
- CI fails when a production module introduces a direct order submit outside the allowlist.

### B04. Use Binance symbol filters for all quantities and prices

Affected: `binance_api/bapi.py`, `binance_api/bapi_placeorder.py`, Binance provider,
`providers/quantity.py`.

Problem:
- Quantity and price use hard-coded decimal rounding in low-level paths.
- Mechanics contains a hard-coded 100 USDC notional threshold that conflates exchange
  minimum with business policy.
- Symbol metadata lookup and normalization are split across legacy and common paths.

Plan:
- Cache and validate `LOT_SIZE`, `MARKET_LOT_SIZE`, `PRICE_FILTER`, `MIN_NOTIONAL` or
  `NOTIONAL`, precision, and applicability-to-market flags.
- Quantize with decimal arithmetic and floor in the financially safe direction.
- Keep exchange minimum and configurable business minimum as distinct fields and reasons.
- Return the complete quantity decision: requested, balance cap, policy cap, fee cap,
  exchange cap, final quantity, and refusal reason.
- Fail closed on missing/stale/inconsistent exchange metadata for live submission.

Gate:
- Boundary/property tests for every enabled Binance symbol, including exact step/tick,
  one unit below minimum, maximum, fee-adjusted BUY, and free-balance SELL.

### B05. Validate financial inputs before policy and mechanics

Affected: common quantity, guard, regime, provider, and Binance mechanics code.

Problem:
- Not every path proves price, quantity, balances, fees, thresholds, timestamps, and
  configuration are finite, positive, and in expected units.
- `None`, empty results, zero, NaN, and infinity can have different meanings but are
  sometimes collapsed.

Plan:
- Centralize strict finite-value parsing and unit validation.
- Represent unavailable balance separately from a real zero balance.
- Treat stale price, unavailable account data, malformed filters, and clock uncertainty as
  explicit refusal states, with configurable zero-balance backoff only for real zero.
- Validate configuration at startup, including percentages, windows, fractions, sleeps,
  attempts, and queue/file limits.

Gate:
- Table-driven tests for None/empty/zero/negative/NaN/infinity/stale/wrong-unit values.

### B06. Smart cancel/replace must reconcile before new exposure

Affected: `adjust_price_and_cancel_opposite`, Binance cancel helpers, common executor.

Problem:
- A failed cancellation can currently be logged while placement continues, creating
  duplicate or crossed exposure.
- Cancel acknowledgment is not equivalent to terminal cancellation when fills race.

Plan:
- Convert replace into an explicit state machine: snapshot, cancel request, terminal
  reconciliation, effective remaining quantity, replacement.
- Recompute balance and quantity after cancel/fill races.
- Fail closed when order ownership or terminal state is ambiguous.

Gate:
- Race tests for fill-before-cancel, partial-fill-during-cancel, cancel timeout, already
  canceled, and unknown order.

## P0 - Bot-specific lifecycle defects

### B07. `monitortrades.py` outcome and balance semantics

Problems:
- `_place_guarded` reports success after calling `inst.place` without checking its result.
- Free-balance retrieval converts unavailable/exception state to zero.
- `can_trade` is assigned in one branch but does not control later actions; the global
  recent-trade gate is therefore misleading.
- Hard-TP cooldown is in memory only and begins on accepted placement, not confirmed fill.
- Position statistics depend on normalized order history whose fill/partial-fill semantics
  must be explicit.

Plan:
- Consume the typed lifecycle outcome and update cooldown only on the chosen confirmed
  lifecycle state.
- Separate account-data unavailable from zero, enforce the intended global trade gate,
  and persist/reconcile hard-TP intent and cooldown.
- Define whether position statistics use fills only, executed quantity of orders, or both;
  use one common fill ledger.

### B08. `assetguardian.py` campaign reconciliation

Problems:
- A tier can be marked complete on accepted placement even if it never fills.
- Campaign state does not retain enough order identity/executed quantity to reconcile a
  restart, cancellation, or partial fill.
- Account balance list failure can look like an empty portfolio.

Plan:
- Persist tier intent, client/order IDs, requested and executed values, and lifecycle.
- Advance a tier according to explicit fill policy; reconcile at startup and each cycle.
- Use typed account snapshot status and locked/free balances consistently.

### B09. `tradeall.py` restart-safe signal and order ownership

Problems:
- Trend fire state and per-trend counters are primarily in memory; restart can re-fire.
- Accepted placement can be treated as confirmed execution.
- Multiple overlapping trigger branches require a single deterministic decision record.

Plan:
- Persist decision ID, signal version/time, chosen branch, intent, order identity, and
  terminal result.
- Deduplicate by decision/intent, not only time cooldown.
- Separate observation/shadow signals from live execution by typed command boundaries.

Status snapshot (2026-08-23): immediate behavior-neutral hardening is implemented:
tradeall preserves its existing common persistent retry policy, rejects invalid side/price and stale
price snapshots before execution, validates unsafe runtime configuration, deduplicates
coordinator symbols and supports deterministic coordinator shutdown. Accepted submissions
are documented as throttling events rather than fills. Persisting trend/decision identity
and reconciling accepted/partial/terminal orders across restart remains open.

### B10. Binance trailing stop must verify execution

Problems:
- Binance adapter `execute_sell` and `execute_rebuy` return success unconditionally after
  calling the common facade.
- Core then mutates/removes rebuy state as if execution succeeded.
- Balance API failure may be represented by an empty list.
- Peak/rebuy state is atomic but not locked for multiple process writers and is not a
  durable order lifecycle journal.

Plan:
- Return true only for the explicitly selected normalized lifecycle result.
- Persist pending action before submit and reconcile on restart.
- Use typed balance snapshot and single-instance plus file-lock guarantees.
- Distinguish dry-run simulation transitions from live state transitions.

Status snapshot (2026-08-23): immediate fail-closed hardening is implemented.
The Binance adapter now advances sell/re-buy state only when the guarded facade
returns an accepted order with an `orderId`; refused or queued attempts preserve
the peak/re-buy state. Empty/invalid balance snapshots, non-finite prices and unsafe
sell-fraction/runtime inputs are rejected. The existing common persistent retry
policy and MARKET crash/re-buy behavior remain unchanged. Full fill reconciliation
is still open: venue acceptance is not a fill, and completing this safely requires a
persisted pending-action journal plus startup reconciliation before any resubmission.

### B11. `rtrade.py` residual hardening

Plan:
- Re-run complete pair-state invariant audit after shared outcome/filter changes.
- Prove BUY->SELL and SELL->BUY symmetry, multiple rounds on the same symbol, deterministic
  IDs, startup reconciliation, partial fills, fees, net quantity, insufficient-funds
  backoff, hard-stop escalation, and bounded state/log data.
- Preserve caller-owned retry; do not merge rtrade pair intents into the generic queue.
- Confirm sleep/event scheduling has no spin loops and all worker/thread shutdown paths
  are bounded and restart-safe.

## P1 - Retry, history, and account APIs

### B12. Replace generic retry claim-before-submit with leased intent state

Affected: `order_retry.py`, `order_retry_worker.py`.

Problems:
- Claim removes a record before submit; worker crash loses it.
- Deduplication by symbol+side can merge unrelated module/strategy intents.
- Queue records do not guarantee the originating signal remains valid.
- Queue atomic rename lacks power-loss durability.

Plan:
- Add owner, strategy, intent ID, client order ID, signal expiry, and validation callback ID.
- Use pending/inflight lease states; recover expired leases after reconciliation.
- Deduplicate by owner+intent, never by symbol+side alone.
- Revalidate strategy validity and exchange truth immediately before submission.
- Keep the worker disabled until these properties and migration behavior are proven.

Status snapshot (2026-08-23): crash-safe leased claims, fsync-backed queue mutation,
finite input validation, deterministic client-order IDs per revision, and Binance
pre/post-submit reconciliation are implemented. A concurrent refresh cannot be deleted
by completion of an older revision. The operator explicitly enabled the worker; common
price/profit/balance/trend/cooldown guards are re-evaluated. Owner/strategy/intent-ID
deduplication and reconstruction of the originating strategy signal remain open, so the
generic queue must not be used by caller-owned state machines such as rtrade.

### B13. One normalized order/fill history API

Affected: `bapi_allorders.py`, `bapi_trades.py`, cache managers, Binance provider,
`providers/order_lifecycle.py`, guard and bot consumers.

Problems:
- Legacy order and fill functions expose different field names and semantics.
- Empty list can mean no records or API failure.
- Some timestamp pagination can skip fills sharing a millisecond boundary; another path
  already uses `fromId` safely.
- Dead/experimental functions remain and confuse the source of truth.
- Daily limits depend on whether canceled orders, accepted orders, or fills are counted.

Plan:
- Select one paginated fill source and one normalized order-status source.
- Return typed query status plus records; standardize milliseconds internally.
- Define daily limit semantics explicitly and use executed quote/quantity where intended.
- Remove dead variants after parity tests and cache migration checks.

### B14. Typed Binance account and cancellation APIs

Affected: `binance_api/bapi.py`, Binance provider.

Problems:
- Balance/open-order calls often return empty containers on exceptions.
- Cancellation helpers return booleans and lose terminal status and race information.
- `normalize_quantity` can receive missing limits and uses float floor arithmetic.
- Module import installs a global SIGINT handler, coupling library import to process control.

Plan:
- Return typed snapshots with `ok/stale/error`, server timestamp, and records.
- Normalize cancel results through common lifecycle and reconcile terminal state.
- Move signal ownership to executable entry points.
- Retire duplicate normalization in favor of common exchange-filter logic.

### B15. Binance client resilience and rate-limit observability

Affected: `binance_api/bapi_client.py`, REST callers, WebSocket adapter.

Plan:
- Verify HTTP retry library configuration never retries non-idempotent Binance requests,
  including SDK-specific request methods.
- Parse and expose weight/order-count headers; use bounded adaptive backoff for 429/418.
- Add circuit state for time sync failure, persistent auth errors, and stale server time.
- Add request correlation without logging credentials or signed query material.
- Ensure shutdown stops time-sync and WebSocket threads in batch/tests.

## P1 - Cache, persistence, concurrency, and bounds

### B16. Single-writer and merge-safe cache persistence

Affected: `cacheManager.py`, `tradeCacheManager.py`, cache factory/singletons.

Problems:
- Atomic rename prevents partial reads but concurrent writers remain last-writer-wins.
- JSONL append and metadata updates are not protected by a cross-process writer lock.
- Freshness guard alone cannot merge disjoint updates with equal/overlapping timestamps.
- Corrupt JSONL lines are silently skipped without quarantine/metrics.

Plan:
- Declare and enforce one writer per cache, with read-only instances structurally unable
  to save; alternatively lock and merge under a versioned schema.
- Lock append/compact/rotate/meta as one transaction boundary.
- Track schema version, writer ID, sequence, checksum, corruption count, and recovery.
- Exercise simultaneous append/compact/resync/rotation and process-kill tests.

### B17. Tighten cache and memory bounds

Affected: cache managers, price windows, trend manager, state trackers, alert caches,
execution audit caches, provider REST caches.

Plan:
- Replace the broad two-year/1GB defaults with per-dataset retention derived from actual
  consumer lookback and backup policy.
- Bound symbols, subscribers, pending commands, history lists, state maps, ID maps, and
  error/dedup maps; use TTL+LRU only when both recency and size are relevant.
- Ensure eviction cannot remove an active financial intent or unresolved order.
- Add runtime gauges and tests that exceed every configured bound.

### B18. Durable writes for critical state; atomic writes for derived state

Affected: pair store, guardian state, trailing state, retry queue, strategy state,
`priceAnalysis.py`, cache metadata.

Problems:
- Several atomic replacements do not fsync the file and containing directory.
- `priceAnalysis.json` is written directly and readers can observe truncation.
- Shared `.tmp` names in some helpers can conflict across processes.

Plan:
- Classify files as critical recovery state, rebuildable cache, or observational log.
- Use durable atomic helper for critical state and unique atomic helper for derived JSON.
- Validate and migrate schema; quarantine corrupt state and rebuild only when safe.

### B19. WebSocket event integrity and polling fallback

Affected: `binance_api/bapi_ws.py`, WS bridge, cache managers.

Plan:
- Track connection health independently from last execution event; an account with no
  orders must not be declared stale solely because no event arrived.
- Verify subscription acknowledgment, sequence/order update monotonicity, duplicate event
  handling, reconnect resync, and event-vs-REST conflict resolution.
- Bound command queues and subscriber lists; expose reconnect count and last successful
  authenticated heartbeat.
- Poll authoritative REST after reconnect before accepting WS-derived terminal state.

## P1 - Strategy guard and market data semantics

### B20. Clarify bypass policy in `Instrument.place`

Problem:
- `bypass_profit_guard` also bypasses quantity policy/weight adjustment, while its name
  suggests a narrower exception. Daily limit, cooldown, and trend wait still apply.

Plan:
- Replace boolean bypasses with an explicit policy profile containing individually named
  controls: profit, quantity/weight, daily limit, cooldown, trend wait, and emergency exit.
- Define which controls may block exposure reduction. Emergency exits must still obey
  exchange filters, available balance, idempotency, and audit.

### B21. Guard history and cooldown correctness

Affected: `order_guard.py`, cooldown locks, Binance order history.

Plan:
- Namespace cooldown and lock keys by venue, account, symbol, side, and policy as needed.
- Normalize timestamp units and source freshness.
- Define anti-spam and daily-cap accounting from normalized lifecycle states.
- Validate profit-reference selection for partial fills, fees, and multiple lots.
- Decide and test fail-closed behavior for missing history versus an actual first trade.

### B22. Common market-regime input without hidden execution changes

Affected: market regime modules and consumers in Binance bots.

Plan:
- Keep classification separate from execution policy; publish horizon-specific result,
  confidence, evidence freshness, source coverage, disagreement, and fallback reason.
- Do not substitute another asset as equivalent evidence; cross-asset signals may only be
  an explicitly weighted contextual feature.
- Backtest every consumer-specific policy before enabling it; preserve existing strategy
  behavior until promotion gates pass.

## P2 - Monitoring, alerts, logs, and disaster recovery

### B23. Health supervision must test progress, not only PID state

Affected: `fleet_supervisor.sh`, healthcheck/watchdog scripts, systemd and cron artifacts.

Plan:
- Give every active trading process a heartbeat/progress contract: last completed cycle,
  last successful external read, unresolved intents, queue depth, and last error class.
- Detect live-but-stuck threads and stale logs, not only missing/stopped PIDs.
- Bound restart rate with backoff and an escalation state; reconcile before restart when a
  process may have submitted an order.
- Keep auto-remediation actions idempotent and auditable.

### B24. Logging and alert resource policy

Affected: `log.py`, logging configuration, `alertnotifiers.py`, execution audit, cron
rotation and retention.

Problems:
- Default log size allowance is very large for a fleet and must be justified per file.
- Alert and JSONL write paths require consistent locking, rotation, redaction, and failure
  metrics.
- Market-alert cleanup thread is started with `None` before the optional new-coin monitor
  exists, so it never cleans that monitor's old-coin state.

Plan:
- Define per-log max size, age, archive count, compression, disk-watermark behavior, and
  secret redaction.
- Bound alert dedup state, retry count, batch size, pending delivery, and local alert file.
- Start cleanup after dependencies are constructed or make the dependency dynamic.
- Alert once per incident state transition to prevent notification storms.

### B25. Disaster-recovery verification

Affected: `systemd/`, cron snapshots, backup/restore scripts, VPN/DNS artifacts, secrets
inventory, PROD and DEV/backtest profiles.

Plan:
- Maintain versioned, secret-free service/cron/network templates and a manifest of secret
  names and restore locations; never store token values in Git.
- Back up critical state, cache databases, audit logs, and configuration with checksums,
  encryption, retention, and verified copies on DEV and local WSL/Windows.
- Automate a clean-machine restore rehearsal that stops before live trading, then validates
  branch/commit, dependencies, timers, permissions, VPN/DNS, data freshness, and dry-run.
- Require explicit live-enable and exchange reconciliation after restoration.

## P2 - Cleanup and maintainability

### B26. Remove dead and contradictory legacy code safely

Candidates: legacy order/fill fetch variants, old normalization/weight functions,
unused imports and globals, archived monitor logic, duplicate provider-specific balance
calculations, and unreachable returns.

Plan:
- Generate a call graph and runtime import inventory before removal.
- Lock intended behavior with characterization tests.
- Remove in small commits and repeat direct-submit and balance-calculation scans.
- Keep one canonical implementation per provider-specific mechanic and one common policy
  implementation above providers.

### B27. Configuration ownership and reload policy

Plan:
- Build a schema listing every environment/config key, owner module, type, bounds,
  default, secret classification, runtime reload behavior, and deployment profile.
- Eliminate duplicate or contradictory defaults across code and files.
- Validate enabled symbols are USDC-quoted where the Binance strategy requires it; retain
  generic quote parsing only in provider-neutral infrastructure.
- Fail startup for unsafe invalid live configuration; keep diagnostic tools import-safe.

## Test and verification backlog

### Mandatory new suites

- Normalized order lifecycle contract across Binance and shared providers.
- Binance exchange-filter property tests with captured metadata fixtures.
- Unknown-submit-outcome and deterministic-client-ID reconciliation.
- Partial-fill accounting including commission asset and quote quantity.
- Cancel/replace races and balance changes between preflight and submit.
- Multi-process cache append/compact/rotate and critical-state crash durability.
- Bot restart recovery for rtrade, assetguardian, tradeall, monitortrades, and trailing.
- Healthcheck live-but-stuck, restart storm, stale cache, and reconciliation-before-restart.
- Resource-bound tests for every queue, map, list, subscriber set, log, and cache file.
- Configuration schema and PROD/DEV disaster-recovery dry-run tests.

### Existing suites to retain as gates

- Binance mechanics, provider executor, quantity decision, instrument guards, lifecycle,
  cooldown, execution audit, retry worker, ownership inventory, WebSocket bridge.
- rtrade pair/store/threading/trend tests.
- assetguardian, monitortrades financial characterization, tradeall observation/window,
  trailing stop, price/trend, cache manager, watchdog, migration, and import-safety tests.
- Full suite on DEV/backtest after every shared-layer batch.

## Completeness matrix

| Area | Files/modules reviewed | Planned batch |
|---|---|---|
| Binance client and time sync | `binance_api/bapi_client.py` | B15 |
| Binance balances, price, open orders, cancel | `binance_api/bapi.py` | B05, B14 |
| Binance placement mechanics | `binance_api/bapi_placeorder.py` | B03-B06 |
| Binance fills and order history | `binance_api/bapi_allorders.py`, `bapi_trades.py` | B13 |
| Binance WebSocket | `binance_api/bapi_ws.py`, WS bridge | B19 |
| Shared execution facade | `instrument.py`, `providers/market_api.py`, strategy executor | B01-B06 |
| Quantity and balances | `providers/quantity.py`, provider adapters | B04-B05 |
| Lifecycle and audit | order lifecycle, execution audit, order ID context | B01-B02 |
| Profit/daily/cooldown guards | `order_guard.py`, `lock/` | B20-B21 |
| Generic retry | `order_retry.py`, `order_retry_worker.py` | B12 |
| rtrade | bot, pair strategy, pair store, coordinator | B11 |
| assetguardian | bot and campaign state | B08 |
| tradeall | bot, trend state, price window | B09 |
| monitortrades | bot and position calculations | B07 |
| Binance trailing | adapter and shared trailing core | B10 |
| Cache and trend writers | `cacheManager.py`, `tradeCacheManager.py`, `priceAnalysis.py` | B16-B18 |
| Market alerts | alert config, fetcher/checker/notifier/discovery | B17, B24 |
| Resource bounds and logs | log, JSON/JSONL/cache/alert structures | B17-B18, B24 |
| Supervision | fleet launcher, healthcheck, watchdogs | B23 |
| Runtime configuration | `procs.conf`, env/conf/instrument configuration | B27 |
| DR and backup | systemd, cron, backup/restore, VPN/DNS manifests | B25 |
| Tests and verification tools | `tests/`, `verify_tools/` | all gates |

## Completion definition

The plan is exhausted only when every batch is either implemented and verified or is
explicitly rejected with a recorded financial/operational rationale; all P0 items are
closed before P1, all shared-layer changes pass every active bot's contract tests, the
direct-submit inventory contains only approved mechanics, restart simulations reconcile
without duplicate/lost orders, all persistent and in-memory structures have tested
bounds, a clean-machine DR rehearsal passes, and PROD observation confirms no lifecycle,
balance, rate-limit, stale-cache, watchdog, or resource-regression alerts.
