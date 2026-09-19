#!/usr/bin/env bash
# The single source for the DEV/backtest profile used by the shell runners.

RUNNER_DIR="${RUNNER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEV_PROFILE="${DEV_BACKTEST_ENV:-$RUNNER_DIR/dev_backtestconfig.env}"
[ -r "$DEV_PROFILE" ] || { echo "Missing DEV profile: $DEV_PROFILE" >&2; return 1; }

set -a
source "$DEV_PROFILE"
set +a

for name in DEV_HOST DEV_PORT DEV_USER DEV_PATH DEV_CODE_BRANCH BACKTEST_PROPOSALS_BRANCH; do
  [ -n "${!name:-}" ] || { echo "Invalid DEV profile: missing $name" >&2; return 1; }
done
