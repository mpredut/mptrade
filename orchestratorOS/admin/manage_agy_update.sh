#!/usr/bin/env bash
# manage_agy_update.sh — Updater & Lifecycle Manager for Antigravity CLI (agy)
# Modes:
#   --manual (default when run manually): checks/applies agy update, then performs
#            a clean daemon cycle (stop -> start --name prod-vm).
#   --cron : checks for updates and only restarts the daemon if a new version was applied.
#   --force: forces daemon stop -> start regardless of update status.
#   --status: displays daemon status and version.
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

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [agy_update] $*"
}

AGY_BIN="$(command -v agy || true)"
if [ -z "$AGY_BIN" ] && [ -x "$HOME/.local/bin/agy" ]; then
    AGY_BIN="$HOME/.local/bin/agy"
fi

if [ -z "$AGY_BIN" ] || [ ! -x "$AGY_BIN" ]; then
    log "ERROR: agy binary not found in PATH or ~/.local/bin/agy" >&2
    exit 1
fi

ACTION="${1:-}"

if [ "$ACTION" = "--status" ]; then
    log "Binary: $AGY_BIN ($("$AGY_BIN" --version 2>/dev/null || echo 'unknown'))"
    "$AGY_BIN" remote-control status || true
    exit 0
fi

log "=== Starting Antigravity CLI update cycle ==="
log "Using binary: $AGY_BIN (current: $("$AGY_BIN" --version 2>/dev/null || echo 'unknown'))"

UPDATE_OUTPUT="$("$AGY_BIN" update 2>&1 || true)"
echo "$UPDATE_OUTPUT"

restart_daemon_cleanly() {
    log "Performing clean daemon cycle (stop -> start --name '${INSTANCE_NAME}')..."

    # If executing from inside the daemon's own cgroup (e.g. web terminal or active agent),
    # dispatch via systemd-run so cgroup destruction does not kill the restart sequence.
    if grep -q "antigravity-cli-daemon.service" /proc/self/cgroup 2>/dev/null && command -v systemd-run >/dev/null 2>&1; then
        log "Executing inside daemon cgroup. Delegating restart to transient user systemd service..."
        local job_unit="agy-restart-$(date +%s)"
        systemd-run --user --unit="$job_unit" bash -c "
            '$AGY_BIN' remote-control stop || true
            sleep 2
            '$AGY_BIN' remote-control start --name '$INSTANCE_NAME'
        "
        log "Transient restart service queued ($job_unit). The daemon will re-initialize shortly."
    else
        "$AGY_BIN" remote-control stop || true
        sleep 2
        "$AGY_BIN" remote-control start --name "$INSTANCE_NAME"
        sleep 2
        "$AGY_BIN" remote-control status || true
    fi
}

if [ "$ACTION" = "--force" ]; then
    log "Forced daemon restart requested."
    restart_daemon_cleanly
elif [ "$ACTION" = "--cron" ]; then
    if echo "$UPDATE_OUTPUT" | grep -iq "Update successful"; then
        log "Update applied successfully under cron. Restarting daemon..."
        restart_daemon_cleanly
    elif echo "$UPDATE_OUTPUT" | grep -iq "already on the latest version"; then
        log "Already up to date. Cron run complete; no restart needed."
    else
        log "Cron check complete with no update applied."
    fi
else
    # Manual execution (default) — run update and restart daemon as prod-vm
    log "Manual update invoked. Restarting daemon as instance '${INSTANCE_NAME}'..."
    restart_daemon_cleanly
fi

log "=== Antigravity CLI update cycle finished ==="
