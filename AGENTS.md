# Repository Rules
- **Strict English:** All code, commits, and documentation must be in English.
- **Auto-Push:** Automatically push verified commits upstream.
- **Targeted Local Testing & Dev CI:** Run only targeted unit tests for touched files during active tasks. Never run the full 1300-test suite inside agent turns. Full regressions run asynchronously on the DEV machine (`offline/runners/trigger_dev_regression.sh`). Check status via `./offline/runners/check_dev_regression.sh` or `cachedb/dev_regression_status.json`.

# Core Tenets
- **Unification & Compaction:** Consolidate fragmented logic into single linear choke points; eliminate redundancy.
- **Single Source of Truth (SSOT):** Canonical definition and resolution for all domain state, config, and rules.
- **Separation of Concerns (SoC):** Isolate distinct responsibilities and modular boundaries; minimize cross-cutting coupling.
