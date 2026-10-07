#!/usr/bin/env bash
# agy_watchdog.sh — Watchdog and auto-recovery for Antigravity Remote Control daemon
# Checks if the remote-control daemon is active with instance name 'prod-vm'.
# If stopped, inactive, or failed, restarts and registers it as 'prod-vm'.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$ROOT/logs"

# Ensure PATH includes user local bin
export PATH="$HOME/.local/bin:$PATH"

# Ensure systemd user session bus is accessible under cron
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
fi

INSTANCE_NAME="prod-vm"
LOCK_FILE="${XDG_RUNTIME_DIR}/agy_watchdog.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [agy_watchdog] $*"
}

# Acquire lock to prevent concurrent watchdog executions
exec 200>"$LOCK_FILE"
flock -n 200 || { exit 0; }

AGY_BIN="$(command -v agy || true)"
if [ -z "$AGY_BIN" ] && [ -x "$HOME/.local/bin/agy" ]; then
    AGY_BIN="$HOME/.local/bin/agy"
fi

if [ -z "$AGY_BIN" ] || [ ! -x "$AGY_BIN" ]; then
    log "ERROR: agy binary not found in PATH or ~/.local/bin/agy" >&2
    exit 1
fi

is_daemon_active() {
    if command -v systemctl >/dev/null 2>&1; then
        if systemctl --user is-active --quiet antigravity-cli-daemon; then
            return 0
        fi
    fi
    # Secondary check: verify via agy remote-control status
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
            -H "Title: AGY Watchdog ($(hostname))" \
            -d "$msg" \
            "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
    fi
}

MODE="${1:-}"

if [ "$MODE" = "--check" ]; then
    if is_daemon_active; then
        echo "active"
        exit 0
    else
        echo "inactive"
        exit 1
    fi
fi

if [ "$MODE" = "--restart" ] || ! is_daemon_active; then
    if [ "$MODE" = "--restart" ]; then
        log "Manual restart requested for instance '${INSTANCE_NAME}'..."
    else
        log "WARNING: Antigravity remote-control daemon is DOWN/INACTIVE. Triggering restart as '${INSTANCE_NAME}'..."
    fi

    # Start and register with the specified instance name
    START_OUT="$("$AGY_BIN" remote-control start --name "$INSTANCE_NAME" 2>&1 || true)"
    echo "$START_OUT"
    sleep 2

    if is_daemon_active; then
        log "SUCCESS: Antigravity daemon is now ACTIVE as '${INSTANCE_NAME}'."
        if [ "$MODE" != "--restart" ]; then
            send_alert "Antigravity remote-control daemon was found down and successfully revived as '${INSTANCE_NAME}'."
        fi
        exit 0
    else
        log "ERROR: Failed to restart Antigravity remote-control daemon. Output: $START_OUT" >&2
        send_alert "CRITICAL: Failed to revive Antigravity daemon (${INSTANCE_NAME}) on $(hostname)."
        exit 2
    fi
else
    # Quiet healthy exit for cron; print if --verbose or interactive terminal
    if [ "$MODE" = "--verbose" ] || [ -t 1 ]; then
        log "Antigravity daemon is active and healthy (instance: ${INSTANCE_NAME})."
    fi
    exit 0
fi
