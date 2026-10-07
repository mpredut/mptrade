#!/usr/bin/env bash
# manage_agy.sh — Unified manager for Antigravity CLI (agy) & remote-control daemon
# Commands:
#   update       (default) Check/apply agy update, then stop & start daemon as 'prod-vm'
#   watchdog     Verify daemon is active; if down/inactive, revive registered as 'prod-vm'
#   cron-update  Check for updates; restart daemon only if a new version was applied
#   restart      Restart daemon registered as 'prod-vm'
#   status       Display daemon status and binary version
#   stop         Stop the daemon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$ROOT/logs"

# Ensure PATH includes user local bin
export PATH="$HOME/.local/bin:$PATH"

# Ensure systemd user session bus is accessible under cron / detached subshells
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
fi

INSTANCE_NAME="prod-vm"
LOCK_FILE="${XDG_RUNTIME_DIR}/manage_agy.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [manage_agy] $*"
}

AGY_BIN="$(command -v agy || true)"
if [ -z "$AGY_BIN" ] && [ -x "$HOME/.local/bin/agy" ]; then
    AGY_BIN="$HOME/.local/bin/agy"
fi

if [ -z "$AGY_BIN" ] || [ ! -x "$AGY_BIN" ]; then
    log "ERROR: agy binary not found in PATH or ~/.local/bin/agy" >&2
    exit 1
fi

is_active() {
    if command -v systemctl >/dev/null 2>&1; then
        if systemctl --user is-active --quiet antigravity-cli-daemon; then
            return 0
        fi
    fi
    local status_output
    status_output="$("$AGY_BIN" remote-control status 2>&1 || true)"
    if echo "$status_output" | grep -qi "Daemon status: active"; then
        return 0
    fi
    return 1
}

send_alert() {
    local msg="$1"
    local topic
    topic=$(grep -hs -m1 '^NTFY_TOPIC_ERROR=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
    [ -n "$topic" ] || topic=$(grep -hs -m1 '^NTFY_TOPIC=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
    if [ -n "$topic" ]; then
        local token
        token=$(grep -hs -m1 '^NTFY_TOKEN=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
        local auth_hdr=()
        [ -n "$token" ] && auth_hdr=(-H "Authorization: Bearer $token")
        curl -fsS -m 10 --retry 2 "${auth_hdr[@]}" \
            -H "Title: AGY Manager ($(hostname))" \
            -d "$msg" \
            "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
    fi
}

do_restart() {
    log "Cycling daemon (stop -> start --name '${INSTANCE_NAME}')..."
    if grep -q "antigravity-cli-daemon.service" /proc/self/cgroup 2>/dev/null && command -v systemd-run >/dev/null 2>&1; then
        log "Executing inside daemon cgroup. Delegating restart to transient user systemd service..."
        local job_unit="agy-restart-$(date +%s)"
        systemd-run --user --unit="$job_unit" bash -c "
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
        local start_out
        start_out="$("$AGY_BIN" remote-control start --name "$INSTANCE_NAME" 2>&1 || true)"
        echo "$start_out"
        sleep 2
        if is_active; then
            log "SUCCESS: Daemon successfully recovered as '${INSTANCE_NAME}'."
            send_alert "Antigravity remote-control daemon was found down and successfully revived as '${INSTANCE_NAME}'."
        else
            log "ERROR: Failed to recover daemon. Output: $start_out" >&2
            send_alert "CRITICAL: Failed to revive Antigravity daemon (${INSTANCE_NAME}) on $(hostname)."
            exit 2
        fi
    else
        if [ -t 1 ] || [ "${1:-}" = "--verbose" ]; then
            log "Daemon is active and healthy (instance: ${INSTANCE_NAME})."
        fi
    fi
}

do_update() {
    local mode="${1:-manual}"
    log "=== Antigravity CLI update check ==="
    log "Using binary: $AGY_BIN (current: $("$AGY_BIN" --version 2>/dev/null || echo 'unknown'))"
    local update_output
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
}

CMD="${1:-update}"
case "$CMD" in
    update|--update)
        do_update "manual"
        ;;
    cron-update|--cron|--cron-update)
        do_update "cron"
        ;;
    watchdog|--watchdog)
        shift 1 2>/dev/null || true
        do_watchdog "$@"
        ;;
    restart|--restart|--force)
        do_restart
        ;;
    status|--status)
        do_status
        ;;
    stop|--stop)
        log "Stopping antigravity remote-control daemon..."
        "$AGY_BIN" remote-control stop || true
        ;;
    -h|--help|help)
        echo "Usage: $0 [update|watchdog|cron-update|restart|status|stop]"
        echo ""
        echo "Commands:"
        echo "  update       (default) Run agy update, then restart daemon as '${INSTANCE_NAME}'"
        echo "  watchdog     Check if daemon is active; revive as '${INSTANCE_NAME}' if down"
        echo "  cron-update  Run update, restart daemon only if an update was applied"
        echo "  restart      Restart daemon registered as '${INSTANCE_NAME}'"
        echo "  status       Show daemon status and CLI version"
        echo "  stop         Stop daemon"
        ;;
    *)
        echo "Unknown command: $CMD. Run '$0 --help' for usage." >&2
        exit 1
        ;;
esac
