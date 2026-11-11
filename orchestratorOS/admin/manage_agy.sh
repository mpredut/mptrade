#!/usr/bin/env bash
# manage_agy.sh - Unified manager for Antigravity CLI (agy) & remote-control daemon
# Commands: update (default) | watchdog | cron-update | restart | ui-sync | status | stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$SCRIPT_DIR/../lib/os_notify.sh"
mkdir -p "$ROOT/logs"

export PATH="$HOME/.local/bin:$PATH"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
fi

INSTANCE_NAME="prod-vm"
LOCK_FILE="${XDG_RUNTIME_DIR}/manage_agy.lock"
UPDATE_PENDING_FILE="${XDG_RUNTIME_DIR}/agy_update_restart_pending"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [manage_agy] $*"; }
send_alert() { [ "${AGY_SUPPRESS_ALERTS:-0}" = "1" ] && return 0; send_os_ntfy "AGY Manager ($(hostname))" "$1" default || true; }

AGY_BIN="$(command -v agy || true)"
[ -z "$AGY_BIN" ] && [ -x "$HOME/.local/bin/agy" ] && AGY_BIN="$HOME/.local/bin/agy"
if [ -z "$AGY_BIN" ] || [ ! -x "$AGY_BIN" ]; then
    log "ERROR: agy binary not found in PATH or ~/.local/bin/agy" >&2
    exit 1
fi

resolve_agy_log() {
    local pid log_file=""
    pid="$(systemctl --user show -p MainPID --value antigravity-cli-daemon 2>/dev/null || true)"
    [ -n "$pid" ] && [ "$pid" -gt 0 ] && [ -d "/proc/$pid" ] && log_file="$(readlink -f "/proc/$pid/fd/1" 2>/dev/null || true)"
    if [ -z "$log_file" ] || [ ! -f "$log_file" ]; then
        log_file="$(readlink -f "$HOME/.gemini/antigravity-cli/cli.log" 2>/dev/null || true)"
    fi
    echo "$log_file"
}

is_active() {
    if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active --quiet antigravity-cli-daemon; then
        return 0
    fi
    "$AGY_BIN" remote-control status 2>&1 | grep -qi "Daemon status: active"
}

is_tunnel_stuck() {
    local log_file min_failures="${AGY_TUNNEL_STUCK_MIN_FAILURES:-3}" status_lines total_count fail_count
    if ! [[ "$min_failures" =~ ^[0-9]+$ ]] || [ "$min_failures" -lt 2 ]; then
        min_failures=3
    fi
    log_file="$(resolve_agy_log)"
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        status_lines="$(grep "Connection status:" "$log_file" 2>/dev/null | tail -n "$min_failures" || true)"
        if [ -n "$status_lines" ]; then
            total_count="$(printf '%s\n' "$status_lines" | grep -c "Connection status:" || true)"
            fail_count="$(printf '%s\n' "$status_lines" | grep -ci "Failed before open" || true)"
            if [ "$total_count" -ge "$min_failures" ] && [ "$fail_count" -ge "$min_failures" ]; then
                return 0
            fi
        fi
    fi
    return 1
}

_MANAGE_AGY_LOCKED=0
acquire_lock() {
    local wait_s="${1:-20}"
    [ "$_MANAGE_AGY_LOCKED" = "1" ] && return 0
    exec 200>"$LOCK_FILE"
    if [ "$wait_s" -le 0 ]; then
        flock -n 200 || return 1
    else
        flock -w "$wait_s" 200 || return 1
    fi
    _MANAGE_AGY_LOCKED=1
}

resolve_python_bin() {
    local python_bin="${PYTHON_BIN:-python3}"
    if ! command -v "$python_bin" >/dev/null 2>&1 && [ ! -x "$python_bin" ]; then
        [ -x "$ROOT/.venv/bin/python3" ] && python_bin="$ROOT/.venv/bin/python3" || python_bin="/usr/bin/python3"
    fi
    echo "$python_bin"
}

resolve_ui_sync_params() {
    local python_bin ssot_vals idle_sec="" min_interval_sec=""
    python_bin="$(resolve_python_bin)"
    ssot_vals="$("$python_bin" -c "import sys; sys.path[:0] = ['$ROOT']; from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals; d = resolve_derived_telemetry_intervals(); print(f\"{int(d['ui_sync_idle_sec'])} {int(d['ui_sync_interval_sec'])}\")" 2>/dev/null | grep -E '^[0-9]+ [0-9]+$' | tail -n 1 || true)"
    read -r idle_sec min_interval_sec <<< "$ssot_vals" || true
    if ! [[ "$idle_sec" =~ ^[0-9]+$ ]] || [ "$idle_sec" -le 0 ]; then
        idle_sec="$(env_get AGY_UI_SYNC_IDLE_SEC 120 | grep -Eo '^[0-9]+' || echo 120)"
    fi
    if ! [[ "$min_interval_sec" =~ ^[0-9]+$ ]] || [ "$min_interval_sec" -le 0 ]; then
        min_interval_sec=720
    fi
    echo "$idle_sec $min_interval_sec"
}

is_interactive_busy() {
    local python_bin params idle_sec="" min_interval_sec="" rc=0
    python_bin="$(resolve_python_bin)"
    params="$(resolve_ui_sync_params)"
    read -r idle_sec min_interval_sec <<< "$params" || true
    if ! [[ "$idle_sec" =~ ^[0-9]+$ ]] || [ "$idle_sec" -le 0 ]; then
        idle_sec=120
    fi
    "$python_bin" "$SCRIPT_DIR/agy_store_fix.py" active-check --idle-sec "$idle_sec" >/dev/null 2>&1 || rc=$?
    [ "$rc" -ne 0 ]
}

do_restart() {
    rm -f "$UPDATE_PENDING_FILE" 2>/dev/null || true
    log "Cycling daemon (stop -> start --name '${INSTANCE_NAME}')..."
    local self_cgroup="${AGY_SELF_CGROUP_FILE:-/proc/self/cgroup}"
    if [ -r "$self_cgroup" ] && grep -q "antigravity-cli-daemon.service" "$self_cgroup" 2>/dev/null && command -v systemd-run >/dev/null 2>&1; then
        log "Executing inside daemon cgroup. Delegating restart to transient user systemd service..."
        local job_unit="agy-restart-$(date +%s)-$$"
        systemd-run --user --unit="$job_unit" bash -c "
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            export XDG_RUNTIME_DIR=\"\${XDG_RUNTIME_DIR:-/run/user/\$(id -u)}\"
            [ -S \"\${XDG_RUNTIME_DIR}/bus\" ] && export DBUS_SESSION_BUS_ADDRESS=\"\${DBUS_SESSION_BUS_ADDRESS:-unix:path=\${XDG_RUNTIME_DIR}/bus}\"
            exec 200>'$LOCK_FILE'
            flock -w 20 200 || true
            sleep 3
            '$AGY_BIN' remote-control stop || true
            for _ in {1..10}; do
                systemctl --user is-active --quiet antigravity-cli-daemon 2>/dev/null || break
                sleep 1
            done
            sleep 2
            if ! '$AGY_BIN' remote-control start --name '$INSTANCE_NAME'; then
                sleep 5
                systemctl --user is-active --quiet antigravity-cli-daemon 2>/dev/null || '$AGY_BIN' remote-control start --name '$INSTANCE_NAME' || true
            fi
        "
        log "Transient restart service queued ($job_unit)."
    else
        acquire_lock 20 || true
        "$AGY_BIN" remote-control stop || true
        for _ in {1..10}; do
            systemctl --user is-active --quiet antigravity-cli-daemon 2>/dev/null || break
            sleep 1
        done
        sleep 2
        if ! "$AGY_BIN" remote-control start --name "$INSTANCE_NAME"; then
            sleep 5
            is_active || "$AGY_BIN" remote-control start --name "$INSTANCE_NAME" || true
        fi
        sleep 2
        "$AGY_BIN" remote-control status || true
    fi
}

do_watchdog() {
    acquire_lock 0 || exit 0

    if command -v systemctl >/dev/null 2>&1; then
        case "$(systemctl --user show -p ActiveState --value antigravity-cli-daemon 2>/dev/null || true)" in
            activating|deactivating|reloading) exit 0 ;;
        esac
    fi

    if ! is_active; then
        log "WARNING: Daemon is DOWN/INACTIVE. Triggering auto-recovery as '${INSTANCE_NAME}'..."
        local start_out start_out_single
        start_out="$("$AGY_BIN" remote-control start --name "$INSTANCE_NAME" 2>&1 || true)"
        if echo "$start_out" | grep -qi "not signed in"; then
            sleep 3
            start_out="$("$AGY_BIN" remote-control start --name "$INSTANCE_NAME" 2>&1 || true)"
        fi
        start_out_single="$(printf '%s' "$start_out" | tr '\r\n' '  ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
        sleep 2
        if is_active; then
            log "SUCCESS: Daemon successfully recovered as '${INSTANCE_NAME}'. Output: ${start_out_single}"
            send_alert "Antigravity remote-control daemon was found down and successfully revived as '${INSTANCE_NAME}'."
        else
            log "ERROR: Failed to recover daemon. Output: ${start_out_single}" >&2
            send_alert "CRITICAL: Failed to revive Antigravity daemon (${INSTANCE_NAME}) on $(hostname)."
            exit 2
        fi
    elif is_tunnel_stuck; then
        if is_interactive_busy; then
            log "WARNING: Daemon tunnel has persistent reconnection failures, but interactive chats/tasks are active; deferring restart until idle."
        else
            log "WARNING: Daemon is running, store is idle, and tunnel is STUCK in reconnection failure. Triggering auto-recovery..."
            do_restart
            send_alert "Antigravity remote-control daemon tunnel was found stuck/broken while idle and was automatically restarted as '${INSTANCE_NAME}'."
        fi
    else
        if [ -t 1 ] || [ "${1:-}" = "--verbose" ] || { [ -f "$ROOT/logs/manage_agy.log" ] && tail -n 5 "$ROOT/logs/manage_agy.log" 2>/dev/null | grep -Eiq "error:|warning:|failed"; }; then
            log "Daemon is active and healthy (instance: ${INSTANCE_NAME})."
        fi
        do_ui_sync
    fi
}

do_update() {
    local mode="${1:-manual}" update_output
    log "=== Antigravity CLI update check ==="
    log "Using binary: $AGY_BIN (current: $("$AGY_BIN" --version 2>/dev/null || echo 'unknown'))"
    update_output="$("$AGY_BIN" update 2>&1 || true)"
    echo "$update_output"

    if [ "$mode" = "cron" ]; then
        if echo "$update_output" | grep -iq "Update successful"; then
            if is_interactive_busy; then
                touch "$UPDATE_PENDING_FILE"
                log "Update applied under cron, but interactive chats/tasks are active; queued deferred restart ($UPDATE_PENDING_FILE)."
            else
                log "Update applied successfully under cron and store is idle. Cycling daemon..."
                do_restart
            fi
        elif echo "$update_output" | grep -iq "already on the latest version"; then
            log "Already up to date. Cron check complete; no restart needed."
        else
            log "Cron check complete with no update applied."
        fi
    else
        log "Manual update invoked. Cycling daemon as instance '${INSTANCE_NAME}'..."
        do_restart
    fi
    log "=== Antigravity CLI update finished ==="
}

do_status() {
    log "Binary: $AGY_BIN ($("$AGY_BIN" --version 2>/dev/null || echo 'unknown'))"
    "$AGY_BIN" remote-control status || true
    local log_file last_status
    log_file="$(resolve_agy_log)"
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        last_status="$(grep "Connection status:" "$log_file" 2>/dev/null | tail -n 1 || true)"
        [ -n "$last_status" ] && log "Tunnel: $last_status"
    fi
}

do_ui_sync() {
    acquire_lock 10 || exit 0
    local python_bin params idle_sec="" min_interval_sec="" rc=0
    python_bin="$(resolve_python_bin)"
    params="$(resolve_ui_sync_params)"
    read -r idle_sec min_interval_sec <<< "$params" || true
    if ! [[ "$idle_sec" =~ ^[0-9]+$ ]] || [ "$idle_sec" -le 0 ]; then
        log "ERROR: ui_sync_idle_sec / AGY_UI_SYNC_IDLE_SEC missing or invalid ('$idle_sec')" >&2
        exit 1
    fi
    is_active || exit 0
    if [ -f "$UPDATE_PENDING_FILE" ]; then
        if ! is_interactive_busy; then
            log "ui-sync: deferred CLI update restart is pending and store is idle (>= ${idle_sec}s); cycling daemon."
            do_restart
            return 0
        fi
    fi
    "$python_bin" "$SCRIPT_DIR/agy_store_fix.py" ui-sync-check --idle-sec "$idle_sec" --min-interval-sec "$min_interval_sec" >/dev/null 2>&1 || rc=$?
    case "$rc" in
        0)  log "ui-sync: store diverged from daemon and idle >= ${idle_sec}s (cadence >= ${min_interval_sec}s); restarting to surface threads."
            do_restart ;;
        10|11|12|3)
            if [ -f "$ROOT/logs/manage_agy.log" ] && tail -n 5 "$ROOT/logs/manage_agy.log" 2>/dev/null | grep -Eiq "error:|warning:|failed"; then
                log "ui-sync: check healthy (rc=$rc, idle_sec=${idle_sec}s)."
            fi ;;
        *)  log "WARNING: ui-sync-check failed (rc=$rc); see ~/agy_store_fix.log" >&2 ;;
    esac
}

CMD="${1:-update}"
case "$CMD" in
    update|--update)                  do_update "manual" ;;
    cron-update|--cron|--cron-update) do_update "cron" ;;
    ui-sync|--ui-sync)                do_ui_sync ;;
    watchdog|--watchdog)              shift 1 2>/dev/null || true; do_watchdog "$@" ;;
    restart|--restart|--force)        do_restart ;;
    status|--status)                  do_status ;;
    stop|--stop)                      log "Stopping antigravity remote-control daemon..."; "$AGY_BIN" remote-control stop || true ;;
    -h|--help|help)                   echo "Usage: $0 [update|watchdog|cron-update|restart|ui-sync|status|stop]" ;;
    *)                                echo "Unknown command: $CMD. Run '$0 --help' for usage." >&2; exit 1 ;;
esac
