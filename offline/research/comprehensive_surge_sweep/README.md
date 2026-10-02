# Explicit-profile surge sweep

The runner loads the shared strategy from one selected versioned venue profile.
It does not load secrets, inherit shell strategy settings, place live orders, or
change live configuration. Experimental candidates override their declared fields;
CURRENT_CONFIG retains the selected profile unchanged.

## Invocation

Run from the repository root, replacing FEE_PERCENTAGE with an explicit per-leg
fee assumption:

    .venv/bin/python offline/research/comprehensive_surge_sweep/run_deep_surge_sweep.py \
      --config kraken/config.env --fee-pct FEE_PERCENTAGE \
      --data-dir offline/results/kraken_continuous_grid/data \
      --out-dir offline/results/surge_sweep --workers 3

The runner does not infer fees from a venue or account. An optional --initial-cash
overrides the configured effective allocation. Without that override, percentage
sizing is respected. Each asset replay has its own cash balance.

All five expected *_240m.csv datasets must exist with at least 60 parsed bars.
Missing data or a failed candidate aborts publication of a new report. Existing
reports are not deleted on failure; a failed run does not refresh them.

## Configuration migration

StratParams.from_env now requires every consumed financial setting, even for
disabled features. Missing, empty numeric, malformed, and non-finite values raise
an error identifying the setting. An explicit dictionary can be passed without
reading or modifying the process environment.

Both versioned venue profiles explicitly retain three previously implicit values:

    STRAT_STOP_LOSS_PCT=0
    STRAT_REENTRY_PEAK_RELATIVE=false
    STRAT_REENTRY_PULLBACK_PCT=1.5

All previously effective parameters are preserved, including the upstream surge
and bounce calibration. Booleans must use true or false; old 1/yes overrides are
rejected. Review private .env and service overrides before deployment.

Custom full profiles and historical config-only profiles, including those passed
to offline/runners/spot_profile_review.py, must explicitly supply all current
settings. Historical snapshots must not silently inherit today's financial policy
to pass validation. Use a reviewed, complete profile or the stored strategy_params
snapshot through the financial benchmark's --params-report interface.

An intentionally empty STRAT_TP_TRANCHES still disables tranches. A malformed
nonempty specification now raises instead of silently selecting a different exit.

## Report contract and limits

Schema version 2 records the base profile, per-candidate parameters, config path,
cash, fees, and replay cadence. The descriptive beats_baseline_both_halves field
replaces two_window_verified; pnl_per_drawdown_point replaces the incorrectly
named calmar_ratio. External consumers must update their field names. Historical
results and baselines are not rewritten.

- Comparing the same grid on both halves is not held-out walk-forward selection.
- Four-hour samples cannot validate a five-minute price guard.
- Asset PnLs use independent cash accounts, not one shared portfolio wallet.
- Fees are explicit assumptions, not verified live account fees.

The surge-window and surge-arming logic, grid ranges, and live financial policy
are outside this refactor. Upstream strategy corrections from 916d444 are retained,
including the fixed surge policy and expanded experiment grid. Their financial
validation remains separate work. Reports do not constitute approval for automatic
promotion to live.
