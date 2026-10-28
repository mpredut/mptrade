#!/usr/bin/env bash
# The single source for the DEV/backtest profile used by the shell runners.

RUNNER_DIR="${RUNNER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEV_PROFILE="${DEV_BACKTEST_ENV:-$RUNNER_DIR/dev_backtest.env}"
[ -r "$DEV_PROFILE" ] || { echo "Missing DEV profile: $DEV_PROFILE" >&2; return 1; }

set -a
source "$DEV_PROFILE"
set +a

for name in DEV_HOST DEV_PORT DEV_USER DEV_PATH DEV_CODE_BRANCH BACKTEST_PROPOSALS_BRANCH; do
  [ -n "${!name:-}" ] || { echo "Invalid DEV profile: missing $name" >&2; return 1; }
done

[ -f "$RUNNER_DIR/../../orchestratorOS/lib/backup_sync_common.sh" ] && source "$RUNNER_DIR/../../orchestratorOS/lib/backup_sync_common.sh"

export BACKTEST_MODE=1
export DISABLE_EXTERNAL_NOTIFICATIONS=1
export DISABLE_GEMINI_LLM=1
