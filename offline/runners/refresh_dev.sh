#!/usr/bin/env bash
# refresh_dev.sh — brings the BACKTEST machine (dev) up to date with prod via two channels:
#   1. CODE: `git pull --ff-only` on dev (code travels through GitHub, not this script).
#   2. DATA (cachedb, untracked by git): rsync prod -> dev.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
# shellcheck source=orchestratorOS/lib/backup_sync_common.sh
source "$RUNNER_DIR/../../orchestratorOS/lib/backup_sync_common.sh"

exec 9>"/tmp/refresh_dev.lock"
if ! flock -w 300 9; then
  echo "[refresh_dev $(date '+%F %T')] another refresh_dev has run for >300s — skipping"
  exit 0
fi

REPO_ROOT="${BINANCE_REPO_ROOT:-$ROOT}"
SSH="ssh -o BatchMode=yes -p $DEV_PORT"

echo "[refresh_dev $(date '+%F %T')] code: git pull --ff-only on dev"
PULL_OUTPUT=$($SSH "$DEV_USER@$DEV_HOST" "cd ~/$DEV_PATH && if flock -n /tmp/dev_regression.lock flock -w 45 /tmp/dev_ci_watcher.lock env GIT_SSH_COMMAND='ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2' timeout 45 git pull --ff-only origin $DEV_CODE_BRANCH; then :; else echo 'Regression currently in flight; deferred git pull until lock is released.'; fi" 2>&1) || {
    echo "[refresh_dev $(date '+%F %T')] WARN: git pull on dev timed out or failed: $PULL_OUTPUT" >&2
}
echo "$PULL_OUTPUT" | sed 's/^/  /'
if echo "$PULL_OUTPUT" | grep -q "Updating "; then
    echo "[refresh_dev $(date '+%F %T')] code updated on DEV -> launching background regression"
    $SSH "$DEV_USER@$DEV_HOST" "nohup bash -c 'cd ~/$DEV_PATH && ./offline/runners/run_dev_regression.sh' </dev/null >/dev/null 2>&1 &" || true
fi

echo "[refresh_dev $(date '+%F %T')] data: rsync cachedb/ prod -> dev"
rsync_cachedb "$REPO_ROOT/cachedb/" "$DEV_USER@$DEV_HOST:$DEV_PATH/cachedb/" \
    --exclude 'dev_regression_status.json' -e "$SSH" | sed 's/^/  /'

echo "[refresh_dev $(date '+%F %T')] done"
