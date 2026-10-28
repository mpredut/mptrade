#!/usr/bin/env bash
# dev_ci_watcher.sh — automated regression watcher on DEV (cron every 2 minutes).
# Polls origin/main cheaply (git fetch); if remote advanced beyond HEAD, pulls and launches run_dev_regression.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../orchestratorOS/lib/env_common.sh
source "$SCRIPT_DIR/../../orchestratorOS/lib/env_common.sh"
cd "$ROOT"

BRANCH="${DEV_CODE_BRANCH:-main}"
exec 9>"/tmp/dev_ci_watcher.lock"
flock -n 9 || exit 0

timeout 15 git fetch -q origin "$BRANCH" 2>/dev/null || exit 0
LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "[dev_ci_watcher $(date '+%F %T')] Remote update detected ($LOCAL -> $REMOTE)."
    if flock -n /tmp/dev_regression.lock true 2>/dev/null; then
        git checkout -- offline/research/*_results.json 2>/dev/null || true
        git pull --ff-only origin "$BRANCH" || { echo "[dev_ci_watcher $(date '+%F %T')] ERROR: git pull --ff-only failed" >&2; exit 1; }
        echo "[dev_ci_watcher $(date '+%F %T')] Git pull successful. Launching regression..."
    else
        echo "[dev_ci_watcher $(date '+%F %T')] Regression currently in flight; queueing run_dev_regression.sh..."
    fi
    nohup "$SCRIPT_DIR/run_dev_regression.sh" </dev/null >/dev/null 2>&1 &
fi
