#!/usr/bin/env bash
# run_dev_regression.sh — runs the full test suite regression on DEV,
# records atomic execution status in cachedb/dev_regression_status.json,
# and emits ntfy alerts on failure.
#
# Runs on DEV (or locally). Thread-safe via flock mutex.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MUTEX="/tmp/dev_regression.lock"
exec 8>"$MUTEX"
WAS_QUEUED=false
if ! flock -n 8; then
    echo "[dev_regression $(date '+%F %T')] Another regression is already running — waiting for lock..."
    WAS_QUEUED=true
    if ! flock -w 600 8; then
        echo "[dev_regression $(date '+%F %T')] ERROR: Timed out waiting for regression lock" >&2
        exit 1
    fi
fi

# Locate Python / Pytest environment
if [ -x "$REPO_ROOT/.venv/bin/pytest" ]; then
    PYTEST_BIN="$REPO_ROOT/.venv/bin/pytest"
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/myenv/bin/pytest" ]; then
    PYTEST_BIN="$REPO_ROOT/myenv/bin/pytest"
    PYTHON_BIN="$REPO_ROOT/myenv/bin/python"
elif [ -x "$REPO_ROOT/venv/bin/pytest" ]; then
    PYTEST_BIN="$REPO_ROOT/venv/bin/pytest"
    PYTHON_BIN="$REPO_ROOT/venv/bin/python"
elif command -v pytest >/dev/null 2>&1; then
    PYTEST_BIN="$(command -v pytest)"
    PYTHON_BIN="$(command -v python3)"
else
    echo "[dev_regression $(date '+%F %T')] ERROR: pytest not found in .venv, myenv, venv, or PATH" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/cachedb"
LOG_FILE="$REPO_ROOT/logs/dev_regression.log"
STATUS_JSON="$REPO_ROOT/cachedb/dev_regression_status.json"

# Pull latest code safely inside the mutex so mid-run pytest imports are never torn
git pull --ff-only origin "${DEV_CODE_BRANCH:-main}" >/dev/null 2>&1 || true

COMMIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
COMMIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
COMMIT_MSG="$(git log -1 --pretty=%s 2>/dev/null || echo "unknown")"

if [ "$WAS_QUEUED" = true ] && [ -f "$STATUS_JSON" ]; then
    LAST_COMMIT_AND_STATUS="$("$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(f"{d.get(\"commit\",\"\")}:{d.get(\"status\",\"\")}")' "$STATUS_JSON" 2>/dev/null || true)"
    if [ "$LAST_COMMIT_AND_STATUS" = "${COMMIT_HASH}:PASSED" ]; then
        echo "[dev_regression $(date '+%F %T')] Commit $COMMIT_SHORT already tested and PASSED while queued — skipping duplicate run"
        exit 0
    fi
fi

START_TS="$(date +%s)"
START_ISO="$(date -Iseconds)"

echo "[dev_regression $START_ISO] Starting regression on commit $COMMIT_SHORT: $COMMIT_MSG" | tee -a "$LOG_FILE"

# Clear backtest/replay environment overrides (e.g. when called from run_overnight_master_suite.sh)
# so unit tests for guards, notifications, and GeminiClient execute normally.
unset BACKTEST_MODE REPLAY_MODE DISABLE_EXTERNAL_NOTIFICATIONS DISABLE_GEMINI_LLM

# Run pytest capturing exit code and output
set +e
"$PYTEST_BIN" tests/ -q > "$LOG_FILE.run" 2>&1
RC=$?
set -e

END_TS="$(date +%s)"
DURATION=$((END_TS - START_TS))
cat "$LOG_FILE.run" >> "$LOG_FILE"

PREV_STATUS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$STATUS_JSON" 2>/dev/null || true)"

# Parse pytest results via python
REG_LOG_PATH="$LOG_FILE.run" \
REG_RC="$RC" \
REG_DURATION="$DURATION" \
REG_COMMIT_HASH="$COMMIT_HASH" \
REG_COMMIT_SHORT="$COMMIT_SHORT" \
REG_COMMIT_MSG="$COMMIT_MSG" \
REG_START_TS="$START_TS" \
REG_START_ISO="$START_ISO" \
REG_STATUS_JSON="$STATUS_JSON" \
"$PYTHON_BIN" - <<'EOF' | tee -a "$LOG_FILE"
import json
import os
import re
import sys
import time

log_path = os.environ["REG_LOG_PATH"]
rc = int(os.environ.get("REG_RC") or 0)
duration = int(os.environ.get("REG_DURATION") or 0)
commit_hash = os.environ.get("REG_COMMIT_HASH", "")
commit_short = os.environ.get("REG_COMMIT_SHORT", "")
commit_msg = os.environ.get("REG_COMMIT_MSG", "")
start_ts = int(os.environ.get("REG_START_TS") or 0)
start_iso = os.environ.get("REG_START_ISO", "")
status_json_path = os.environ["REG_STATUS_JSON"]

passed = 0
failed = 0
errors = 0
warnings = 0

try:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    # Example match: 1297 passed, 4 warnings in 137.33s
    # Example match: 2 failed, 1295 passed in 140s
    p_m = re.search(r'(\d+)\s+passed', content)
    if p_m:
        passed = int(p_m.group(1))
    
    f_m = re.search(r'(\d+)\s+failed', content)
    if f_m:
        failed = int(f_m.group(1))
        
    e_m = re.search(r'(\d+)\s+errors?', content)
    if e_m:
        errors = int(e_m.group(1))
        
    w_m = re.search(r'(\d+)\s+warnings?', content)
    if w_m:
        warnings = int(w_m.group(1))
except Exception as e:
    print(f"Warning parsing test output: {e}", file=sys.stderr)

status = "PASSED" if rc == 0 and failed == 0 and errors == 0 else "FAILED"

status_data = {
    "status": status,
    "exit_code": rc,
    "commit": commit_hash,
    "commit_short": commit_short,
    "commit_msg": commit_msg,
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "warnings": warnings,
    "duration_sec": duration,
    "started_at": start_iso,
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "ts": int(time.time()),
}

tmp_out = f"{status_json_path}.tmp"
with open(tmp_out, "w", encoding="utf-8") as f:
    json.dump(status_data, f, indent=2)

os.replace(tmp_out, status_json_path)
print(f"[dev_regression] Status: {status} | Passed: {passed} | Failed: {failed} | Duration: {duration}s")
EOF

# Notification on failure or recovery
if [ "$RC" -ne 0 ]; then
    echo "[dev_regression $(date '+%F %T')] FAILED on commit $COMMIT_SHORT ($RC)" | tee -a "$LOG_FILE" >&2
    # If os_notify.sh exists, notify ntfy
    if [ -f "$REPO_ROOT/orchestratorOS/lib/os_notify.sh" ]; then
        source "$REPO_ROOT/orchestratorOS/lib/os_notify.sh"
        send_os_ntfy "DEV CI Regression FAILED" "Commit $COMMIT_SHORT failed: $COMMIT_MSG (exit code: $RC). Duration: ${DURATION}s." "high" "ntfy-error-1978" || true
    fi
else
    echo "[dev_regression $(date '+%F %T')] PASSED on commit $COMMIT_SHORT in ${DURATION}s" | tee -a "$LOG_FILE"
    if [ "$PREV_STATUS" = "FAILED" ] && [ -f "$REPO_ROOT/orchestratorOS/lib/os_notify.sh" ]; then
        source "$REPO_ROOT/orchestratorOS/lib/os_notify.sh"
        send_os_ntfy "DEV CI Regression RECOVERED" "Commit $COMMIT_SHORT passed all tests: $COMMIT_MSG. Duration: ${DURATION}s." "default" "ntfy-error-1978" || true
    fi
fi

rm -f "$LOG_FILE.run"
exit "$RC"
