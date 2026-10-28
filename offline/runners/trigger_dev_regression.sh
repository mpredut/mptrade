#!/usr/bin/env bash
# trigger_dev_regression.sh — triggers code pull and full test regression on DEV from PROD.
# Usage: ./offline/runners/trigger_dev_regression.sh [--wait]
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"

LOCAL_SHORT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
echo "[trigger_dev_ci $(date '+%F %T')] Contacting DEV ($DEV_HOST:$DEV_PORT) to pull and run regression..."
dev_remote_git_pull 2>&1 | sed 's/^/  [dev] /'

if [ "${1:-}" == "--wait" ]; then
    echo "[trigger_dev_ci $(date '+%F %T')] Running regression on DEV (synchronous / waiting)..."
    set +e
    dev_ssh "cd ~/$DEV_PATH && ./offline/runners/run_dev_regression.sh" 2>&1 | sed 's/^/  [dev] /'
    RC=${PIPESTATUS[0]}
    set -e
    dev_sync_regression_status
    echo "[trigger_dev_ci $(date '+%F %T')] Completed (exit=$RC) and synced status to prod cachedb/system/dev_regression_status.json"
    exit "$RC"
else
    dev_launch_regression_bg
    echo "[trigger_dev_ci $(date '+%F %T')] Regression triggered asynchronously in background on DEV for commit $LOCAL_SHORT."
    echo "Check status anytime with: ./offline/runners/check_dev_regression.sh"
fi
