# Refactor continuation handoff

Last updated: 2026-09-07

Latest local handoff: [Binance instrument onboarding](BINANCE_INSTRUMENT_ONBOARDING.md).
This supersedes the older symbol-onboarding assumptions below. Production was
unavailable during this batch; deployment and live ARB state still need verification.

Use this document only if the remaining work is still relevant when the refactor resumes.
Re-check `main`, production behavior, tests and current call sites before implementing it.

## Governing principle

The purpose of the refactor is less active code and more genuinely shared mechanics.
Do not add an abstraction unless it removes concrete duplication or closes a verified
correctness/recovery gap. Do not change strategy signals, thresholds, budgets or retry
policy as part of a mechanical refactor.

## Completed

- atomic, fail-closed runtime persistence and explicit state ownership;
- mandatory configuration and complete credential-pair resolution;
- common spot-DCA mechanics and typed `accepted` / `refused` / `unknown` submissions;
- policy-preserving informed reconciliation and T212 common submit mechanics;
- explicit read-only Kraken and Hyperliquid clients;
- common read-only active-intent index and healthcheck reporting;
- common Binance exchange-filter normalization;
- static allowlist gate for direct low-level submit/cancel boundaries;
- common native order-ID extraction across providers;
- one Binance low-level dispatch implementation with compatibility wrappers;
- obsolete `providers/tracked_order.py` compatibility shim removed;
- English-only active source/comments and Unix-oriented repository policy.

Latest relevant commits at this handoff:

- `75f57e7 Remove the obsolete tracked-order compatibility shim`
- `7a868c1 Centralize Binance filters and execution boundaries`
- `b70a3b2 Add a read-only active intent index`
- `9b29e1d Type submission outcomes and reuse them in T212`

Last complete suite for the main refactor batch: `1173 passed`, `324 subtests passed`.
After the final shim removal, the focused lifecycle/configuration suite passed `22/22`.

## Re-evaluate before continuing

1. Search for current provider duplication after the latest English-only commits.
   Extract only identical validation, status or error mechanics; preserve real venue
   differences.
2. Consider declarative intent fields such as `cancel_origin`, signal validity and
   partial-fill policy only when they replace duplicated strategy code. Merely adding
   fields or a framework is not a useful refactor.
3. Keep the active-intent index observational. Do not make it an authoritative ledger
   until owner-by-owner characterization proves that code will become smaller and that
   financial recovery behavior remains unchanged.
4. Move or delete additional legacy code only after a fresh call-graph check. Deletion is
   acceptable when the file contains no unique business logic and Git history preserves it.
5. Update production verification separately from local test status; never infer that a
   pushed commit is deployed or running.

## Resume checklist

1. `git status --short` and compare local `main` with `origin/main`.
2. Read this handoff plus `docs/ORDER_LIFECYCLE_CENTRALIZATION.md` and
   `docs/STATE_OWNERSHIP.md`.
3. Run the direct-order inventory and the focused provider/lifecycle tests.
4. Inventory exact duplicated blocks and estimate lines removed versus lines added.
5. Implement one bounded batch only when the expected active-code balance is favorable.
6. Run targeted tests, the complete suite, `git diff --check`, then commit in English.
7. Rebase safely over concurrent remote work and push without force.
