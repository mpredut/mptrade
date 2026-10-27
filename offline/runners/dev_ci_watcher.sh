#!/usr/bin/env bash
# dev_ci_watcher.sh — automated regression watcher on DEV.
# Polls origin/main cheaply (git fetch). If remote has advanced beyond local HEAD,
# pulls cleanly and launches run_dev_regression.sh.
#
# Runs on DEV (typically scheduled via crontab every 2 minutes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BRANCH="${DEV_CODE_BRANCH:-main}"
MUTEX="/tmp/dev_ci_watcher.lock"
exec 9>"$MUTEX"
if ! flock -n 9; then
    # Another watcher run is active
    exit 0
fi

# Fetch remote refs quietly with a short network timeout
timeout 15 git fetch -q origin "$BRANCH" 2>/dev/null || exit 0

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "[dev_ci_watcher $(date '+%F %T')] Remote update detected ($LOCAL -> $REMOTE)."
    if flock -n /tmp/dev_regression.lock true 2>/dev/null; then
        git checkout -- offline/research/*_results.json 2>/dev/null || true
        if git pull --ff-only origin "$BRANCH"; then
            echo "[dev_ci_watcher $(date '+%F %T')] Git pull successful. Launching regression..."
            nohup "$SCRIPT_DIR/run_dev_regression.sh" </dev/null >/dev/null 2>&1 &
        else
            echo "[dev_ci_watcher $(date '+%F %T')] ERROR: git pull --ff-only failed" >&2
        fi
    else
        echo "[dev_ci_watcher $(date '+%F %T')] Regression currently in flight; queueing run_dev_regression.sh to pull after lock release..."
        nohup "$SCRIPT_DIR/run_dev_regression.sh" </dev/null >/dev/null 2>&1 &
    fi
fi
