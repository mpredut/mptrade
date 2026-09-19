#!/usr/bin/env bash
# run_backtest_cycle.sh — a complete backtest cycle on DEV: it runs the pilot in
# --propose mode (it applies nothing) and publishes the proposals to the git branch. Runs
# on DEV. Triggered from prod (trigger_backtest_dev.sh through ssh) or by hand.
#
# The data (cachedb) and the code (git pull) are brought up to date by refresh_dev.sh RUN
# ON PROD before this cycle is triggered — here we only run and publish.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BINANCE_REPO_ROOT:-${ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}}"
source "$REPO_ROOT/env_common.sh"
ONLY="${PILOT_ONLY:-}"            # empty = every key; e.g. "maxage,hardtp"
cd "$REPO_ROOT"

echo "[cycle $(date '+%F %T')] pilot --propose (only='${ONLY:-all}')"
args=(--propose)
[ -n "$ONLY" ] && args+=(--only "$ONLY")
"$PYTHON_BIN" offline/research/monitortrades_backtest/scheduled_pilot.py "${args[@]}"

echo "[cycle $(date '+%F %T')] publish proposals to git"
"$REPO_ROOT/offline/runners/publish_proposals.sh"

echo "[cycle $(date '+%F %T')] done"
