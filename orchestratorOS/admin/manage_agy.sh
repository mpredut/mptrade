#!/usr/bin/env bash
# manage_agy.sh — Unified manager for Antigravity CLI (agy) & remote-control daemon
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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [manage_agy] $*"; }
send_alert() { send_os_ntfy "AGY Manager ($(hostname))" "$1" default || true; }

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
    local log_file last_status
    log_file="$(resolve_agy_log)"
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        last_status="$(grep "Connection status:" "$log_file" 2>/dev/null | tail -n 1 || true)"
        if [ -n "$last_status" ] && echo "$last_status" | grep -qi "Failed before open"; then
            local fail_count
            fail_count="$(grep -c "Connection status: Failed before open" <(tail -n 30 "$log_file" 2>/dev/null) || true)"
            [ "$fail_count" -ge 2 ] && return 0
        fi
    fi
    return 1
}

do_restart() {
    log "Cycling daemon (stop -> start --name '${INSTANCE_NAME}')..."
    if grep -q "antigravity-cli-daemon.service" /proc/self/cgroup 2>/dev/null && command -v systemd-run >/dev/null 2>&1; then
        log "Executing inside daemon cgroup. Delegating restart to transient user systemd service..."
        local job_unit="agy-restart-$(date +%s)"
        systemd-run --user --unit="$job_unit" bash -c "
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            export XDG_RUNTIME_DIR=\"\${XDG_RUNTIME_DIR:-/run/user/\$(id -u)}\"
            [ -S \"\${XDG_RUNTIME_DIR}/bus\" ] && export DBUS_SESSION_BUS_ADDRESS=\"\${DBUS_SESSION_BUS_ADDRESS:-unix:path=\${XDG_RUNTIME_DIR}/bus}\"
            sleep 3
            '$AGY_BIN' remote-control stop || true
            sleep 2
            '$AGY_BIN' remote-control start --name '$INSTANCE_NAME'
        "
        log "Transient restart service queued ($job_unit)."
    else
        "$AGY_BIN" remote-control stop || true
        sleep 2
        "$AGY_BIN" remote-control start --name "$INSTANCE_NAME"
        sleep 2
        "$AGY_BIN" remote-control status || true
    fi
}

do_watchdog() {
    exec 200>"$LOCK_FILE"
    flock -n 200 || exit 0

    if ! is_active; then
        log "WARNING: Daemon is DOWN/INACTIVE. Triggering auto-recovery as '${INSTANCE_NAME}'..."
        local start_out start_out_single
        start_out="$("$AGY_BIN" remote-control start --name "$INSTANCE_NAME" 2>&1 || true)"
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
        log "WARNING: Daemon is running but tunnel is STUCK in reconnection failure. Triggering auto-recovery..."
        do_restart
        send_alert "Antigravity remote-control daemon tunnel was found stuck/broken and was automatically restarted as '${INSTANCE_NAME}'."
    elif [ -t 1 ] || [ "${1:-}" = "--verbose" ]; then
        log "Daemon is active and healthy (instance: ${INSTANCE_NAME})."
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
            log "Update applied successfully under cron. Cycling daemon..."
            do_restart
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
    exec 200>"$LOCK_FILE"
    flock -w 10 200 || exit 0
    local idle_sec rc=0
    idle_sec="$(env_get AGY_UI_SYNC_IDLE_SEC)"
    if ! [[ "$idle_sec" =~ ^[0-9]+$ ]] || [ "$idle_sec" -le 0 ]; then
        log "ERROR: AGY_UI_SYNC_IDLE_SEC missing or invalid in config.env ('$idle_sec')" >&2
        exit 1
    fi
    is_active || exit 0
    /usr/bin/python3 "$SCRIPT_DIR/agy_store_fix.py" ui-sync-check --idle-sec "$idle_sec" >/dev/null 2>&1 || rc=$?
    case "$rc" in
        0)  log "ui-sync: store diverged from daemon and idle >= ${idle_sec}s; restarting to surface threads."
            do_restart ;;
        10|11|12|3) ;;
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
