# Repository Rules
- **Strict English:** All code, commits, and documentation must be in English.
- **Auto-Push:** Automatically push verified commits upstream.
- **Targeted Local Testing & Dev CI:** Run only targeted unit tests for touched files during active tasks. Never run the full 1300-test suite inside agent turns. Full regressions run asynchronously on the DEV machine (`offline/runners/trigger_dev_regression.sh`). Check status via `./offline/runners/check_dev_regression.sh` or `cachedb/dev_regression_status.json`.

# Core Tenets
- **Unification & Compaction:** Consolidate fragmented logic into single linear choke points; eliminate redundancy.
- **Single Source of Truth (SSOT):** Canonical definition and resolution for all domain state, config, and rules.
- **Separation of Concerns (SoC):** Isolate distinct responsibilities and modular boundaries; minimize cross-cutting coupling.
- **Holistic & General Solutions Over Point Fixes:**
  - Always solve classes of problems at the architectural choke point (e.g., transport layer, order gateway, state layer) rather than scattering per-caller `if/else` patches across modules.
  - Reject ad-hoc special cases, band-aid flags, and hardcoded magic percentages/timers when a single unified invariant or dynamic market-relative mechanism (e.g., relative to volatility/ATR/regime) can govern the behavior cleanly.
  - Before adding a new branch, cooldown, or state tracker, check whether existing complexity can be deleted or unified instead. Simpler, linear code with fewer states is strictly preferred over deeply nested special-case handling.
  - When a bug is found in one caller or venue, audit the common boundary so the fix prevents the entire category of bug everywhere at once.
