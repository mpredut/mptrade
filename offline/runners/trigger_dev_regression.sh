#!/usr/bin/env bash
# trigger_dev_regression.sh — triggers code pull and full test regression on DEV from PROD.
#
# Usage:
#   ./offline/runners/trigger_dev_regression.sh          # Fire-and-forget (instant, async)
#   ./offline/runners/trigger_dev_regression.sh --wait   # Wait for completion & sync status to prod
#
# Runs on PROD (uses SSH to dev).
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"

REPO_ROOT="${BINANCE_REPO_ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5 -p $DEV_PORT"

WAIT_MODE=false
if [ "${1:-}" == "--wait" ]; then
    WAIT_MODE=true
fi

LOCAL_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
LOCAL_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"

echo "[trigger_dev_ci $(date '+%F %T')] Contacting DEV ($DEV_HOST:$DEV_PORT) to pull and run regression..."

# Step 1: Pull latest code on DEV only if no regression is currently mid-run
# (if a regression holds /tmp/dev_regression.lock, run_dev_regression.sh will pull after acquiring the lock)
$SSH "$DEV_USER@$DEV_HOST" "cd ~/$DEV_PATH && if flock -n /tmp/dev_regression.lock git pull --ff-only origin $DEV_CODE_BRANCH; then :; else echo 'Regression currently in flight; deferred git pull until lock is released.'; fi" 2>&1 | sed 's/^/  [dev] /'

# Step 2: Trigger regression on dev
if [ "$WAIT_MODE" = true ]; then
    echo "[trigger_dev_ci $(date '+%F %T')] Running regression on DEV (synchronous / waiting)..."
    set +e
    $SSH "$DEV_USER@$DEV_HOST" "cd ~/$DEV_PATH && ./offline/runners/run_dev_regression.sh" 2>&1 | sed 's/^/  [dev] /'
    RC=${PIPESTATUS[0]}
    set -e

    # Sync status back to prod regardless of pass/fail
    mkdir -p "$REPO_ROOT/cachedb"
    scp -P "$DEV_PORT" "$DEV_USER@$DEV_HOST:~/$DEV_PATH/cachedb/dev_regression_status.json" "$REPO_ROOT/cachedb/dev_regression_status.json" >/dev/null 2>&1 || true
    echo "[trigger_dev_ci $(date '+%F %T')] Completed (exit=$RC) and synced status to prod cachedb/dev_regression_status.json"
    exit "$RC"
else
    # Async background trigger on DEV (full nohup subshell detachment)
    $SSH "$DEV_USER@$DEV_HOST" "nohup bash -c 'cd ~/$DEV_PATH && ./offline/runners/run_dev_regression.sh' </dev/null >/dev/null 2>&1 &"
    echo "[trigger_dev_ci $(date '+%F %T')] Regression triggered asynchronously in background on DEV for commit $LOCAL_SHORT."
    echo "Check status anytime with: ./offline/runners/check_dev_regression.sh"
fi
