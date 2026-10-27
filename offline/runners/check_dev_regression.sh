#!/usr/bin/env bash
# check_dev_regression.sh — inspects DEV CI regression status from PROD.
# Fetches the latest cachedb/dev_regression_status.json from DEV (if reachable)
# and formats a clean one-line or multi-line summary.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"

REPO_ROOT="${BINANCE_REPO_ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=3 -p $DEV_PORT"

mkdir -p "$REPO_ROOT/cachedb"
STATUS_FILE="$REPO_ROOT/cachedb/dev_regression_status.json"

# Check if DEV host is reachable
DEV_REACHABLE=true
if ! $SSH "$DEV_USER@$DEV_HOST" "true" >/dev/null 2>&1; then
    DEV_REACHABLE=false
fi

IS_RUNNING=false
if [ "$DEV_REACHABLE" = true ]; then
    # Try to sync latest status from dev
    scp -o BatchMode=yes -o ConnectTimeout=3 -P "$DEV_PORT" \
        "$DEV_USER@$DEV_HOST:~/$DEV_PATH/cachedb/dev_regression_status.json" \
        "$STATUS_FILE" >/dev/null 2>&1 || true
    if ! $SSH "$DEV_USER@$DEV_HOST" "flock -n /tmp/dev_regression.lock true 2>/dev/null"; then
        IS_RUNNING=true
    fi
else
    # DEV unreachable: check if running locally
    if ! flock -n /tmp/dev_regression.lock true 2>/dev/null; then
        IS_RUNNING=true
    fi
fi

if [ ! -f "$STATUS_FILE" ]; then
    if [ "$IS_RUNNING" = true ]; then
        echo "[DEV CI] Regression is currently RUNNING..."
    elif [ "$DEV_REACHABLE" = false ]; then
        echo "[DEV CI] DEV machine ($DEV_HOST:$DEV_PORT) is unreachable and no local regression status is available."
    else
        echo "[DEV CI] No regression status available yet."
    fi
    exit 0
fi

STATUS_FILE="$STATUS_FILE" IS_RUNNING="$IS_RUNNING" python3 - <<'EOF'
import json
import os
import time

status_file = os.environ["STATUS_FILE"]
is_running = os.environ.get("IS_RUNNING") == "true"

try:
    with open(status_file, "r") as f:
        data = json.load(f)
        
    status = data.get("status", "UNKNOWN")
    passed = data.get("passed", 0)
    failed = data.get("failed", 0)
    errors = data.get("errors", 0)
    duration = data.get("duration_sec", 0)
    commit = data.get("commit_short", "unknown")
    commit_msg = data.get("commit_msg", "")
    ts = data.get("ts", 0)
    age_s = int(time.time() - ts) if ts else 0
    age_m = age_s // 60
    
    if is_running:
        print(f"[DEV CI] ⏳ RUNNING: A new regression is currently in progress on DEV (last finished: {status} on {commit} {age_m}m ago).")
    else:
        icon = "✅" if status == "PASSED" else "❌"
        print(f"[DEV CI] {icon} {status} on [{commit}] (\"{commit_msg}\") | Passed: {passed} | Failed: {failed} | Time: {duration}s | Completed {age_m}m ago")
except Exception as e:
    print(f"[DEV CI] Error reading status: {e}")
EOF
