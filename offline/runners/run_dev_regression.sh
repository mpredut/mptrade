#!/usr/bin/env bash
# run_dev_regression.sh — runs the full test suite regression on DEV,
# records atomic execution status in cachedb/system/dev_regression_status.json,
# and emits ntfy alerts on failure/recovery. Thread-safe via flock mutex.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../orchestratorOS/lib/os_notify.sh
source "$SCRIPT_DIR/../../orchestratorOS/lib/os_notify.sh"
cd "$ROOT"

exec 8>"/tmp/dev_regression.lock"
WAS_QUEUED=false
if ! flock -n 8; then
    echo "[dev_regression $(date '+%F %T')] Another regression is already running — waiting for lock..."
    WAS_QUEUED=true
    flock -w 600 8 || { echo "[dev_regression $(date '+%F %T')] ERROR: Timed out waiting for regression lock" >&2; exit 1; }
fi

mkdir -p "$ROOT/logs" "$ROOT/cachedb/system"
LOG_FILE="$ROOT/logs/dev_regression.log"
STATUS_JSON="$ROOT/cachedb/system/dev_regression_status.json"

git checkout -- offline/research/*_results.json 2>/dev/null || true
flock -w 45 /tmp/dev_ci_watcher.lock git pull --ff-only origin "${DEV_CODE_BRANCH:-main}" >/dev/null 2>&1 || true

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
unset BACKTEST_MODE REPLAY_MODE DISABLE_EXTERNAL_NOTIFICATIONS DISABLE_GEMINI_LLM

set +e
"$PYTHON_BIN" -m pytest tests/ -q > "$LOG_FILE.run" 2>&1
RC=$?
set -e

DURATION=$(($(date +%s) - START_TS))
cat "$LOG_FILE.run" >> "$LOG_FILE"
PREV_STATUS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$STATUS_JSON" 2>/dev/null || true)"

REG_LOG_PATH="$LOG_FILE.run" REG_RC="$RC" REG_DURATION="$DURATION" \
REG_COMMIT_HASH="$COMMIT_HASH" REG_COMMIT_SHORT="$COMMIT_SHORT" REG_COMMIT_MSG="$COMMIT_MSG" \
REG_START_ISO="$START_ISO" REG_STATUS_JSON="$STATUS_JSON" \
"$PYTHON_BIN" - <<'EOF' | tee -a "$LOG_FILE"
import json, os, re, sys, time
rc, duration, status_json_path = int(os.environ.get("REG_RC") or 0), int(os.environ.get("REG_DURATION") or 0), os.environ["REG_STATUS_JSON"]
counts = {"passed": 0, "failed": 0, "errors": 0, "warnings": 0}
try:
    content = open(os.environ["REG_LOG_PATH"], "r", encoding="utf-8", errors="replace").read()
    for key, pat in [("passed", r"(\d+)\s+passed"), ("failed", r"(\d+)\s+failed"), ("errors", r"(\d+)\s+errors?"), ("warnings", r"(\d+)\s+warnings?")]:
        m = re.search(pat, content)
        if m:
            counts[key] = int(m.group(1))
except Exception as e:
    print(f"Warning parsing test output: {e}", file=sys.stderr)
status = "PASSED" if rc == 0 and counts["failed"] == 0 and counts["errors"] == 0 else "FAILED"
status_data = {
    "status": status, "exit_code": rc,
    "commit": os.environ.get("REG_COMMIT_HASH", ""), "commit_short": os.environ.get("REG_COMMIT_SHORT", ""), "commit_msg": os.environ.get("REG_COMMIT_MSG", ""),
    **counts, "duration_sec": duration, "started_at": os.environ.get("REG_START_ISO", ""), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ts": int(time.time()),
}
with open(f"{status_json_path}.tmp", "w", encoding="utf-8") as f:
    json.dump(status_data, f, indent=2)
os.replace(f"{status_json_path}.tmp", status_json_path)
print(f"[dev_regression] Status: {status} | Passed: {counts['passed']} | Failed: {counts['failed']} | Duration: {duration}s")
EOF

if [ "$RC" -ne 0 ]; then
    echo "[dev_regression $(date '+%F %T')] FAILED on commit $COMMIT_SHORT ($RC)" | tee -a "$LOG_FILE" >&2
    send_os_ntfy "DEV CI Regression FAILED" "Commit $COMMIT_SHORT failed: $COMMIT_MSG (exit code: $RC). Duration: ${DURATION}s." "high" || true
else
    echo "[dev_regression $(date '+%F %T')] PASSED on commit $COMMIT_SHORT in ${DURATION}s" | tee -a "$LOG_FILE"
    [ "$PREV_STATUS" = "FAILED" ] && send_os_ntfy "DEV CI Regression RECOVERED" "Commit $COMMIT_SHORT passed all tests: $COMMIT_MSG. Duration: ${DURATION}s." "default" || true
fi
rm -f "$LOG_FILE.run"
exit "$RC"
