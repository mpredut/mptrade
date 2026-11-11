#!/usr/bin/env bash
# run_dev_regression.sh -- Unified DEV CI Regression Controller:
#   ./run_dev_regression.sh [run]            -- Execute full pytest suite locally on DEV & record status JSON
#   ./run_dev_regression.sh watch            -- Cron watcher on DEV: poll origin/main & spawn regression on new commits
#   ./run_dev_regression.sh trigger [--wait] -- Trigger DEV regression remotely over SSH from PROD/WSL
#   ./run_dev_regression.sh check            -- Inspect DEV regression status from PROD/WSL/DEV
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
cd "$ROOT"

mkdir -p "$ROOT/logs" "$ROOT/cachedb/system"
LOG_FILE="$ROOT/logs/dev_regression.log"
STATUS_JSON="$ROOT/cachedb/system/dev_regression_status.json"
LOCK="${DEV_REGRESSION_LOCK:-/tmp/dev_regression.lock}"
QUEUE_LOCK="${DEV_REGRESSION_QUEUE_LOCK:-/tmp/dev_regression_queued.lock}"
WATCHER_LOCK="${DEV_CI_WATCHER_LOCK:-/tmp/dev_ci_watcher.lock}"
OVERNIGHT_LOCK="${OVERNIGHT_SUITE_LOCK:-/tmp/mptrade_overnight_master_suite.lock}"

do_watch() {
    exec 9>"$WATCHER_LOCK"
    flock -n 9 || exit 0
    flock -n "$OVERNIGHT_LOCK" true 2>/dev/null || exit 0
    timeout 15 git fetch -q origin "$DEV_CODE_BRANCH" 2>/dev/null || exit 0
    local local_sha remote_sha
    local_sha="$(git rev-parse HEAD)"
    remote_sha="$(git rev-parse "origin/$DEV_CODE_BRANCH")"
    [ "$local_sha" != "$remote_sha" ] || exit 0

    echo "[dev_ci_watcher $(date '+%F %T')] Remote update detected ($local_sha -> $remote_sha)."
    if flock -n "$LOCK" true 2>/dev/null; then
        dev_local_git_sync "$DEV_CODE_BRANCH" || { echo "[dev_ci_watcher $(date '+%F %T')] ERROR: git reset failed" >&2; exit 1; }
        echo "[dev_ci_watcher $(date '+%F %T')] Git reset to origin/$DEV_CODE_BRANCH successful. Launching regression..."
    else
        echo "[dev_ci_watcher $(date '+%F %T')] Regression currently in flight; queueing run_dev_regression.sh..."
    fi
    flock -u 9 2>/dev/null || true
    exec 9>&-
    nohup "$RUNNER_DIR/run_dev_regression.sh" run </dev/null >/dev/null 2>&1 &
}

do_trigger() {
    if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--wait" ]; }; then
        echo "Usage: $0 trigger [--wait]" >&2
        exit 2
    fi
    local local_short rc=0
    local_short="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    echo "[trigger_dev_ci $(date '+%F %T')] Contacting DEV ($DEV_HOST:$DEV_PORT) to pull and run regression..."
    dev_remote_git_pull 2>&1 | sed 's/^/  [dev] /'
    if [ "${1:-}" = "--wait" ]; then
        echo "[trigger_dev_ci $(date '+%F %T')] Running regression on DEV (synchronous / waiting)..."
        dev_ssh "if [[ \"$DEV_PATH\" == /* ]]; then cd \"$DEV_PATH\"; else cd \"\$HOME/$DEV_PATH\"; fi && REG_WAIT_CALLER=1 ./offline/runners/run_dev_regression.sh run" 2>&1 | sed 's/^/  [dev] /' || rc=${PIPESTATUS[0]}
        dev_sync_regression_status
        echo "[trigger_dev_ci $(date '+%F %T')] Completed (exit=$rc) and synced status to cachedb/system/dev_regression_status.json"
        exit "$rc"
    fi
    dev_launch_regression_bg
    echo "[trigger_dev_ci $(date '+%F %T')] Regression triggered asynchronously in background on DEV for commit $local_short."
    echo "Check status anytime with: ./offline/runners/check_dev_regression.sh"
}

do_check() {
    local status_file="$STATUS_JSON" dev_reachable=false is_running=false lock_probe="UNREACHABLE"
    [ ! -f "$status_file" ] && [ -f "$ROOT/cachedb/dev_regression_status.json" ] && status_file="$ROOT/cachedb/dev_regression_status.json"
    lock_probe="$(DEV_SSH_TIMEOUT=3 dev_ssh "flock -n \"$LOCK\" true 2>/dev/null && echo IDLE || echo BUSY" 2>/dev/null || echo UNREACHABLE)"
    if [ "$lock_probe" = "IDLE" ] || [ "$lock_probe" = "BUSY" ]; then
        dev_reachable=true
        dev_sync_regression_status
        [ -f "$STATUS_JSON" ] && status_file="$STATUS_JSON"
        [ "$lock_probe" = "BUSY" ] && is_running=true
    else
        flock -n "$LOCK" true 2>/dev/null || is_running=true
    fi
    if [ ! -f "$status_file" ]; then
        [ "$is_running" = true ] && { echo "[DEV CI] Regression is currently RUNNING..."; exit 0; }
        [ "$dev_reachable" = false ] && { echo "[DEV CI] DEV machine ($DEV_HOST:$DEV_PORT) is unreachable and no local regression status is available."; exit 0; }
        echo "[DEV CI] No regression status available yet."; exit 0
    fi
    STATUS_FILE="$status_file" IS_RUNNING="$is_running" "$PYTHON_BIN" - <<'EOF'
import os, time
from botcore import safe_int
from state_io import read_json_dict
try:
    d = read_json_dict(os.environ["STATUS_FILE"], default={})
    st = d.get("status", "UNKNOWN")
    cm = d.get("commit_short", "unknown")
    msg = d.get("commit_msg", "")
    ts_val = safe_int(d.get("ts"), default=0)
    age = max(0, safe_int(time.time() - ts_val, default=0)) // 60 if ts_val else 0
    if os.environ.get("IS_RUNNING") == "true":
        print(f"[DEV CI] RUNNING: A new regression is currently in progress on DEV (last finished: {st} on {cm} {age}m ago).")
    else:
        icon = "[PASS]" if st == "PASSED" else "[FAIL]"
        print(f"[DEV CI] {icon} {st} on [{cm}] (\"{msg}\") | Passed: {d.get('passed', 0)} | Failed: {d.get('failed', 0)} | Time: {d.get('duration_sec', 0)}s | Completed {age}m ago")
except Exception as e:
    print(f"[DEV CI] Error reading status: {e}")
EOF
}

do_run() {
    exec 8>"$LOCK"
    local was_queued=false
    if ! flock -n 8; then
        if [ "${REG_WAIT_CALLER:-0}" != "1" ]; then
            exec 6>"$QUEUE_LOCK"
            if ! flock -n 6; then
                echo "[dev_regression $(date '+%F %T')] Regression already running and a queued run is already waiting -- coalescing duplicate request."
                exit 0
            fi
        fi
        echo "[dev_regression $(date '+%F %T')] Another regression is already running -- waiting for lock..."
        was_queued=true
        flock -w 600 8 || { echo "[dev_regression $(date '+%F %T')] ERROR: Timed out waiting for regression lock" >&2; exit 1; }
        if [ "${REG_WAIT_CALLER:-0}" != "1" ]; then
            flock -u 6 2>/dev/null || true
            exec 6>&-
        fi
    fi

    local sync_ok=true
    if [ "${SKIP_GIT_SYNC:-0}" != "1" ]; then
        if ! flock -n "$OVERNIGHT_LOCK" true 2>/dev/null; then
            sync_ok=false
            echo "[dev_regression $(date '+%F %T')] Overnight master suite active -- skipping git reset" | tee -a "$LOG_FILE"
        else
            exec 7>"$WATCHER_LOCK"
            if flock -w 45 7; then
                dev_local_git_sync "$DEV_CODE_BRANCH" >/dev/null 2>&1 || sync_ok=false
            else
                sync_ok=false
            fi
            flock -u 7 2>/dev/null || true
            exec 7>&-
        fi
    fi

    local commit_hash commit_short commit_msg start_ts start_iso rc=0 duration prev_status
    commit_hash="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
    commit_short="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    commit_msg="$(git log -1 --pretty=%s 2>/dev/null || echo "unknown")"

    if [ "$was_queued" = true ] && [ "$sync_ok" = true ] && [ -f "$STATUS_JSON" ]; then
        local last_cm="" last_rc="1"
        IFS='|' read -r last_cm last_rc < <("$PYTHON_BIN" -c 'import sys; from botcore import safe_int; from state_io import read_json_dict; d=read_json_dict(sys.argv[1], default={}); raw_rc=d.get("exit_code"); rc=safe_int(raw_rc, default=(0 if d.get("status")=="PASSED" else 1)); cm=str(d.get("commit") or "").strip(); print(f"{cm}|{rc}")' "$STATUS_JSON" 2>/dev/null || echo "|1")
        if [ -n "$last_cm" ] && [ "$last_cm" = "$commit_hash" ]; then
            echo "[dev_regression $(date '+%F %T')] Commit $commit_short already tested while queued -- skipping duplicate run (exit=${last_rc:-0})"
            exit "${last_rc:-0}"
        fi
    fi

    start_ts="$(date +%s)"; start_iso="$(date -Iseconds)"
    echo "[dev_regression $start_iso] Starting regression on commit $commit_short: $commit_msg" | tee -a "$LOG_FILE"
    unset BACKTEST_MODE REPLAY_MODE DISABLE_EXTERNAL_NOTIFICATIONS DISABLE_GEMINI_LLM

    ( exec 8>&-; timeout --kill-after=30 600 "$PYTHON_BIN" -m pytest tests/ -q ) > "$LOG_FILE.run" 2>&1 || rc=$?
    duration=$(($(date +%s) - start_ts))
    cat "$LOG_FILE.run" >> "$LOG_FILE"
    prev_status="$("$PYTHON_BIN" -c 'import sys; from state_io import read_json_dict; print(read_json_dict(sys.argv[1], default={}).get("status",""))' "$STATUS_JSON" 2>/dev/null || true)"

    REG_LOG_PATH="$LOG_FILE.run" REG_RC="$rc" REG_DURATION="$duration" \
    REG_COMMIT_HASH="$commit_hash" REG_COMMIT_SHORT="$commit_short" REG_COMMIT_MSG="$commit_msg" \
    REG_START_ISO="$start_iso" REG_STATUS_JSON="$STATUS_JSON" \
    "$PYTHON_BIN" - <<'EOF' | tee -a "$LOG_FILE"
import os, re, sys, time
from botcore import safe_int
from state_io import atomic_write_json, read_text_file
rc = safe_int(os.environ.get("REG_RC"), default=0)
duration = safe_int(os.environ.get("REG_DURATION"), default=0)
status_json_path = os.environ["REG_STATUS_JSON"]
counts = {"passed": 0, "failed": 0, "errors": 0, "warnings": 0}
try:
    content = read_text_file(os.environ["REG_LOG_PATH"], default="")
    summary_line = ""
    summary_re = re.compile(r"(?:^|[=,\s])\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)(?:\b|$)")
    for raw_line in reversed(content.splitlines()):
        stripped = raw_line.strip()
        if stripped and summary_re.search(stripped) and (" in " in stripped or stripped.startswith("=")):
            summary_line = stripped
            break
    if not summary_line:
        for raw_line in reversed(content.splitlines()):
            stripped = raw_line.strip()
            if stripped and summary_re.search(stripped):
                summary_line = stripped
                break
    for key, pat in [("passed", r"(\d+)\s+passed"), ("failed", r"(\d+)\s+failed"), ("errors", r"(\d+)\s+errors?"), ("warnings", r"(\d+)\s+warnings?")]:
        m = re.search(pat, summary_line)
        if m:
            counts[key] = safe_int(m.group(1), default=0)
except Exception as e:
    print(f"Warning parsing test output: {e}", file=sys.stderr)
status = "PASSED" if rc == 0 and counts["passed"] > 0 and counts["failed"] == 0 and counts["errors"] == 0 else "FAILED"
status_data = {
    "status": status, "exit_code": rc,
    "commit": os.environ.get("REG_COMMIT_HASH", ""), "commit_short": os.environ.get("REG_COMMIT_SHORT", ""), "commit_msg": os.environ.get("REG_COMMIT_MSG", ""),
    **counts, "duration_sec": duration, "started_at": os.environ.get("REG_START_ISO", ""), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ts": safe_int(time.time(), default=0),
}
atomic_write_json(status_json_path, status_data, indent=2)
print(f"[dev_regression] Status: {status} | Passed: {counts['passed']} | Failed: {counts['failed']} | Duration: {duration}s")
EOF

    local failed_tests=""
    if [ "$rc" -ne 0 ] && [ -f "$LOG_FILE.run" ]; then
        failed_tests="$(grep -E '^FAILED ' "$LOG_FILE.run" 2>/dev/null | sed 's/^FAILED //' | head -n 5 | paste -sd ',' - | sed 's/,/, /g' || true)"
    fi
    rm -f "$LOG_FILE.run"
    if [ "$rc" -ne 0 ]; then
        echo "[dev_regression $(date '+%F %T')] FAILED on commit $commit_short ($rc)" | tee -a "$LOG_FILE" >&2
        send_os_ntfy "DEV CI Regression FAILED" "Commit $commit_short failed: $commit_msg (exit code: $rc). Duration: ${duration}s.${failed_tests:+ Failed: $failed_tests}" "high" || true
    else
        echo "[dev_regression $(date '+%F %T')] PASSED on commit $commit_short in ${duration}s" | tee -a "$LOG_FILE"
        [ "$prev_status" = "FAILED" ] && send_os_ntfy "DEV CI Regression RECOVERED" "Commit $commit_short passed all tests: $commit_msg. Duration: ${duration}s." "default" || true
    fi
    exit "$rc"
}

case "${1:-run}" in
    run)            [ "$#" -le 1 ] || { echo "Usage: $0 {run|watch|trigger [--wait]|check}" >&2; exit 2; }; do_run ;;
    watch)          [ "$#" -le 1 ] || { echo "Usage: $0 {run|watch|trigger [--wait]|check}" >&2; exit 2; }; do_watch ;;
    trigger)        shift; do_trigger "$@" ;;
    check|status)   [ "$#" -le 1 ] || { echo "Usage: $0 {run|watch|trigger [--wait]|check}" >&2; exit 2; }; do_check ;;
    *)              echo "Usage: $0 {run|watch|trigger [--wait]|check}" >&2; exit 2 ;;
esac
