#!/usr/bin/env bash
# manage_agy_update.sh — Automatic updater for Antigravity CLI (agy)
# Checks for agy updates and restarts the user daemon only when an update is applied.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

# Ensure PATH includes user local bin
export PATH="$HOME/.local/bin:$PATH"

# Ensure systemd user session bus is accessible when running under cron
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=== Starting Antigravity CLI update check ==="

AGY_BIN="$(command -v agy || true)"
if [ -z "$AGY_BIN" ] && [ -x "$HOME/.local/bin/agy" ]; then
    AGY_BIN="$HOME/.local/bin/agy"
fi

if [ -z "$AGY_BIN" ] || [ ! -x "$AGY_BIN" ]; then
    log "ERROR: agy binary not found in PATH or ~/.local/bin/agy" >&2
    exit 1
fi

log "Using binary: $AGY_BIN ($("$AGY_BIN" --version 2>/dev/null || echo 'version unknown'))"

UPDATE_OUTPUT="$("$AGY_BIN" update 2>&1 || true)"
echo "$UPDATE_OUTPUT"

if echo "$UPDATE_OUTPUT" | grep -iq "Update successful"; then
    log "Update detected and installed successfully."
    if command -v systemctl >/dev/null 2>&1; then
        log "Restarting antigravity-cli-daemon.service..."
        systemctl --user restart antigravity-cli-daemon
        log "antigravity-cli-daemon restarted successfully."
    else
        log "WARNING: systemctl not found; daemon not restarted."
    fi
elif echo "$UPDATE_OUTPUT" | grep -iq "already on the latest version"; then
    log "Already up to date. No service restart needed."
else
    log "Check completed with no update applied."
fi

log "=== Update check finished ==="
