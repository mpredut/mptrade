# Refactor Status

**Reviewed:** 2026-10-04
**Status:** Active summary of implemented architecture and remaining safe work.

## Purpose

This document replaces the archived broad backlog as the concise source of truth
for current refactor work. It records what the code already does, what remains a
small and evidence-based improvement, and which earlier proposals are no longer
planned. It is not authorization to change trading thresholds or deploy a new
financial policy.

## Architecture decisions

- There will be no global or common financial ledger. Durable authority remains
  either in the retry outbox or in strategy-owned campaign, cycle, and position
  state.
- There will be no global or common portfolio-risk engine. Each strategy keeps
  its own budgets and exposure policy, while provider and order safeguards remain
  shared where their mechanics are genuinely identical.
- Semantic deduplication is not planned. Independent intents must not be merged
  because they happen to share a symbol and side. Mechanical idempotency through
  an exact intent, record, client order ID, and venue order ID remains required.
- Market-regime classification is shared evidence, not a fleet-wide strategy.
  Each bot retains its own bounded financial policy and neutral/unknown behavior.

## Current implementation

### Provider-neutral market regime

`market_regime.py` provides the common classification layer:

- canonical snapshot normalization, including the legacy
  `growth_coefficient` input alias;
- typed decisions, reusable placement context, evidence, resolution, and bundle
  objects;
- short- and long-horizon classification from a snapshot or provider OHLC;
- explicit source, age, freshness, fallback, closed-candle, and candle-gap
  evidence;
- bounded positive and negative caching;
- composite execution, balanced, and risk profiles with confidence, conflict,
  conviction, and an explicit unknown state;
- benchmark context that may reinforce or veto local asset evidence but cannot
  manufacture or reverse an asset direction by itself.

`MarketApi` is the composition root for provider routing and exposes regime
decision, evidence resolution, composite bundle, and reusable context APIs. The
runtime short-trend cache is injected at this boundary. Market-regime resolution
in `order_guard` does not import `cacheManager` or the global `MarketApi` instance.

`Instrument.place` resolves one context for a BUY placement and reuses the same
object for the provider reference window and the shared profit guard. A context
passed by a caller is consumed as transient evidence and is not serialized into
the retry outbox; a later retry resolves current evidence. The same context also
reaches the final executable-price profit check on guarded MARKET paths. Legacy
provider hooks that do not accept `regime_context` remain explicitly supported
through signature inspection, so an internal `TypeError` is not mistaken for an
old hook signature. Submission telemetry records the regime, strength, source,
freshness, fallback, and reason when a context is available.

Current consumers are intentionally not identical:

- `tradeall` obtains a reusable placement context from `MarketApi`;
- `rtrade` obtains its short-horizon decision from `MarketApi` and keeps its
  fail-closed sideways-only spread policy;
- the shared spot engine uses `MarketRegimeService` for its long-horizon overlay
  and reuses a completed-bar decision inside one strategy tick.

This is shared interpretation with strategy-owned decisions, not a single global
trading strategy.

### Market intelligence and shadow guards

The `intelligence/` package is a separate decision-support layer. `MarketRegime`
classifies price direction, horizon, freshness, and provenance; market
intelligence combines directional triggers with execution brakes. Its current
pillars are internal price/statistical evidence, external derivatives and market
microstructure evidence, sentiment/LLM evidence, and macro/geopolitical evidence.
`CompositeMarketIntelligence` exposes a typed consolidated evaluation without
owning orders, balances, positions, budgets, or cross-strategy exposure.

The package consolidates reusable Kalman, gradient, volatility, persistence,
survival, parabolic-surge, exhaustion, and noise components. Existing analysis
code now reuses these internal primitives instead of keeping separate
implementations.

The shared order boundary is deliberately narrower than the composite. For BUY
orders, `order_guard.check_intelligence_guards` currently evaluates optional
parabolic history, Weibull trend duration, high-stake Gemini review, and a cached
geopolitical assessment. The versioned modes are `shadow`, so these observations
do not block or resize an order under those defaults. This hook does not call
`CompositeMarketIntelligence`, and it is not a global portfolio-risk coordinator.

Current integration limits are explicit:

- no runtime bot currently consumes `CompositeMarketIntelligence`; the external
  and sentiment composite inputs are not wired into the live order hook;
- `profit_guard` does not forward price history, and placement callers do not
  supply `qty` or `notional_eur` to it, so the parabolic and high-stake Gemini
  checks are not exercised by those paths;
- `profit_guard` consumes the allow/block result but discards `suggested_scale`,
  so a downscale decision does not yet change executable quantity;
- `MarketRegimeContext` does not carry trend duration, so Weibull evaluation may
  fall back to the legacy cache, while geopolitical evaluation consumes only a
  persisted assessment and does not fetch or analyze news in the order path;
- one MARKET placement can call `profit_guard` up to three times. Intelligence
  evidence is not yet captured once and reused, so directly adding remote inputs
  would risk repeated state reads and repeated high-latency reviews;
- the offline intelligence backtest covers a simplified single-asset internal
  strategy. It is not evidence for external, sentiment, macro, or live execution
  behavior.

### Order lifecycle and state ownership

`order_retry.py` contains the reusable outbox and tracked-lifecycle mechanics.
Strategies that own a campaign use their own durable state and
`caller_owns_retry=True`; general `Instrument.place` traffic uses the global
outbox worker. Acceptance remains distinct from a confirmed fill, and venue truth
is used for terminal reconciliation.

`active_intents.py` is a read-only operational index over existing state files.
It is not a ledger, does not own financial state, and has no submit, cancel,
retry, repair, or portfolio-risk authority.

The versioned retry configuration keeps `RETRY_DEDUP=false`. One intent remains one record.
Deterministic IDs prevent replay of that exact record without collapsing separate
financial decisions.

## Regression coverage

Characterization tests cover snapshot normalization, stale/future data, candle
continuity, fallback provenance, composite conflict, bounded caching, injected
resolvers, and single-context reuse across reference and profit guards. Placement
coverage includes legacy provider-hook compatibility, a single hook invocation,
and exclusion of transient regime context from retry serialization. The current
run result belongs to change verification, not this status document.

## Remaining safe work

1. [COMPLETED] Bind a reusable `MarketRegimeContext` to its symbol/provider, trend
   duration, and optional benchmark context, and validate identity and freshness
   via `is_valid_for()`. Guards bypass mismatched or stale contexts and resolve fresh.
2. Resolve intelligence evidence once per placement and reuse a typed result
   across guard checks. Inject trend duration, history, and cached assessments
   through an explicit boundary instead of repeated hidden state reads. Preserve
   current shadow behavior; wiring notional into every guard call could otherwise
   invoke a high-latency review up to three times for one MARKET order.
3. Adopt the shared resolver in another bot only when that bot needs the same
   evidence semantics. Do not force every strategy through one policy or change
   its existing thresholds as part of a mechanical refactor.
4. Remove provider-specific payload parsing only when a duplicated mechanical
   boundary is identified. Keep authentication, precision, endpoint translation,
   and normalized order state in adapters, with signal and financial policy above.

## Experimental strategy work

The composite classifier and multi-horizon evidence are available for controlled
experiments. Any new use for sizing, retry cadence, profit thresholds, trailing
distance, MARKET fallback, or entry/exit selection requires a separately reviewed
policy with bounds, neutral fallback, replay, shadow observation, and explicit
promotion criteria. Cross-asset fallback must remain contextual evidence and must
not silently create a trade in another asset.

## Explicit non-goals

- a global/common ledger;
- a global/common portfolio-risk coordinator;
- semantic intent deduplication;
- automatic fleet-wide activation of the composite classifier;
- changing financial thresholds under the label of refactoring.

## Verification gates

- stale, future, missing, and discontinuous source data resolve explicitly rather
  than being treated as a valid trend;
- fallback selection and provenance remain observable;
- one placement reuses one context and does not persist it in retry state;
- response loss, restart, partial fill, and cancel/fill races do not create a
  second submission for the same mechanical intent;
- provider-specific recovery and each strategy's existing financial behavior stay
  covered by characterization tests;
- deployment and live activation remain separate from code completion.

## Related documents

- [ORDER_LIFECYCLE_CENTRALIZATION.md](ORDER_LIFECYCLE_CENTRALIZATION.md)
- [ORDER_RETRY_ARCHITECTURE.md](ORDER_RETRY_ARCHITECTURE.md)
- [ORDER_INTENT_DEDUP_DESIGN.md](ORDER_INTENT_DEDUP_DESIGN.md)
- [DYNAMIC_MULTI_HORIZON_PROFIT_DESIGN.md](DYNAMIC_MULTI_HORIZON_PROFIT_DESIGN.md)
- [Archived refactor backlog](archive/REFACTOR_AND_SMART_STRATEGY_BACKLOG.md)
