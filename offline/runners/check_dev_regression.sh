#!/usr/bin/env bash
# check_dev_regression.sh — inspects DEV CI regression status from PROD/WSL.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
# shellcheck source=orchestratorOS/lib/backup_sync_common.sh
source "$RUNNER_DIR/../../orchestratorOS/lib/backup_sync_common.sh"

mkdir -p "$ROOT/cachedb/system"
STATUS_FILE="$ROOT/cachedb/system/dev_regression_status.json"
[ ! -f "$STATUS_FILE" ] && [ -f "$ROOT/cachedb/dev_regression_status.json" ] && STATUS_FILE="$ROOT/cachedb/dev_regression_status.json"

DEV_REACHABLE=false
IS_RUNNING=false
if DEV_SSH_TIMEOUT=3 dev_ssh "true" >/dev/null 2>&1; then
    DEV_REACHABLE=true
    dev_sync_regression_status
    [ -f "$ROOT/cachedb/system/dev_regression_status.json" ] && STATUS_FILE="$ROOT/cachedb/system/dev_regression_status.json"
    DEV_SSH_TIMEOUT=3 dev_ssh "flock -n /tmp/dev_regression.lock true 2>/dev/null" || IS_RUNNING=true
else
    flock -n /tmp/dev_regression.lock true 2>/dev/null || IS_RUNNING=true
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

STATUS_FILE="$STATUS_FILE" IS_RUNNING="$IS_RUNNING" "$PYTHON_BIN" - <<'EOF'
import json, os, time
try:
    with open(os.environ["STATUS_FILE"], "r", encoding="utf-8") as f:
        d = json.load(f)
    status = d.get("status", "UNKNOWN")
    commit = d.get("commit_short", "unknown")
    age_m = int(time.time() - d.get("ts", 0)) // 60 if d.get("ts") else 0
    if os.environ.get("IS_RUNNING") == "true":
        print(f"[DEV CI] ⏳ RUNNING: A new regression is currently in progress on DEV (last finished: {status} on {commit} {age_m}m ago).")
    else:
        icon = "✅" if status == "PASSED" else "❌"
        print(f'[DEV CI] {icon} {status} on [{commit}] ("{d.get("commit_msg", "")}") | Passed: {d.get("passed", 0)} | Failed: {d.get("failed", 0)} | Time: {d.get("duration_sec", 0)}s | Completed {age_m}m ago')
except Exception as e:
    print(f"[DEV CI] Error reading status: {e}")
EOF
