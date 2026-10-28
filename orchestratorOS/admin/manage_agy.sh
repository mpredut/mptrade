#!/usr/bin/env bash
# manage_agy.sh — Antigravity CLI daemon manager: watchdog, restart, status, update, ui-sync, clean-logs, clean-sessions.
# Usage: ./manage_agy.sh [watchdog|restart|status|update|ui-sync|clean-logs|clean-sessions]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$SCRIPT_DIR/../lib/os_notify.sh"

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
UID_NUM="$(id -u)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$UID_NUM}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

SERVICE="antigravity-cli-daemon.service"
AGY_DIR="$HOME/.gemini/antigravity"
AGY_CLI_DIR="$HOME/.gemini/antigravity-cli"
LOG_FILE="$ROOT/logs/agy_manager.log"
STATE_FILE="/tmp/agy_watchdog_failures"
RESTART_LOCK="/tmp/agy_restart.lock"
RESTART_TS_FILE="/tmp/agy_last_restart_ts"
MAX_FAILURES=2
COOLDOWN_SEC=180

mkdir -p "$ROOT/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AGY] $*" | tee -a "$LOG_FILE"; }
send_alert() { send_os_ntfy "$1" "$2" "${3:-default}"; }

resolve_agy_log() {
  local pid log_path=""
  pid=$(systemctl --user show -p MainPID --value "$SERVICE" 2>/dev/null || echo 0)
  [ "${pid:-0}" -gt 0 ] && [ -e "/proc/$pid/fd/1" ] && log_path=$(readlink -f "/proc/$pid/fd/1" 2>/dev/null || true)
  { [ -z "$log_path" ] || [ ! -f "$log_path" ]; } && log_path="$AGY_CLI_DIR/cli.log"
  echo "$log_path"
}

is_daemon_running() { systemctl --user is-active --quiet "$SERVICE" 2>/dev/null; }

is_tunnel_stuck() {
  local log_path recent
  log_path="$(resolve_agy_log)"
  [ -f "$log_path" ] || return 1
  recent=$(tail -n 30 "$log_path" 2>/dev/null || true)
  if echo "$recent" | grep -qE "Connection failed, reconnecting|StatusCode\.UNAVAILABLE|Unable to reach the server|Deadline Exceeded|Authentication error"; then
    local last_err last_ok
    last_err=$(echo "$recent" | grep -nE "Connection failed|UNAVAILABLE|Unable to reach|Deadline Exceeded" | tail -1 | cut -d: -f1)
    last_ok=$(echo "$recent" | grep -nE "Listening for|Connected to|Registered instance|Started session|Heartbeat OK" | tail -1 | cut -d: -f1)
    [ -n "${last_err:-}" ] && { [ -z "${last_ok:-}" ] || [ "$last_err" -gt "$last_ok" ]; } && return 0
  fi
  return 1
}

do_restart() {
  local reason="${1:-manual}"
  exec 8>"$RESTART_LOCK"
  flock -n 8 || { log "Restart already in progress, skipping."; return 0; }
  local now last_ts=0
  now=$(date +%s)
  [ -f "$RESTART_TS_FILE" ] && last_ts=$(cat "$RESTART_TS_FILE" 2>/dev/null || echo 0)
  if [ "$reason" != "manual" ] && [ $((now - last_ts)) -lt "$COOLDOWN_SEC" ]; then
    log "Cooldown active ($((now - last_ts))s < ${COOLDOWN_SEC}s), skipping restart."
    return 0
  fi
  log "Restarting $SERVICE (reason: $reason)..."
  systemctl --user stop "$SERVICE" 2>/dev/null || true
  sleep 2
  pkill -9 -f "agy .*daemon" 2>/dev/null || true
  pkill -9 -f "antigravity-cli.*daemon" 2>/dev/null || true
  sleep 1
  systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
  systemctl --user start "$SERVICE" 2>/dev/null || true
  date +%s > "$RESTART_TS_FILE"
  sleep 5
  if is_daemon_running; then
    log "$SERVICE restarted successfully (PID=$(systemctl --user show -p MainPID --value "$SERVICE" 2>/dev/null))."
    echo 0 > "$STATE_FILE"
    return 0
  fi
  log "ERROR: $SERVICE failed to start!"
  send_alert "AGY Daemon Failed ($(hostname))" "antigravity-cli-daemon.service failed to start after restart ($reason)." "high"
  return 1
}

do_watchdog() {
  if ! is_daemon_running; then
    log "WARN: $SERVICE is not active! Triggering immediate restart..."
    do_restart "service_inactive" && send_alert "AGY Daemon Auto-Restarted ($(hostname))" "Service was inactive and has been restarted." "default"
    return
  fi
  if is_tunnel_stuck; then
    local fails=0
    [ -f "$STATE_FILE" ] && fails=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
    fails=$((fails + 1))
    echo "$fails" > "$STATE_FILE"
    log "WARN: Tunnel appears stuck ($fails/$MAX_FAILURES)."
    if [ "$fails" -ge "$MAX_FAILURES" ]; then
      do_restart "tunnel_stuck_${fails}_checks" && send_alert "AGY Tunnel Auto-Healed ($(hostname))" "Tunnel was stuck in reconnect loop ($fails checks) and daemon was restarted." "default"
    fi
  else
    echo 0 > "$STATE_FILE"
  fi
}

do_status() {
  echo "=== Antigravity CLI Daemon Status ($(hostname)) ==="
  echo "Binary   : $(command -v agy 2>/dev/null || echo 'NOT FOUND')"
  echo "Version  : $(agy --version 2>/dev/null || echo 'unknown')"
  echo "Service  : $(systemctl --user is-active "$SERVICE" 2>/dev/null || echo 'inactive') (PID=$(systemctl --user show -p MainPID --value "$SERVICE" 2>/dev/null || echo 0))"
  systemctl --user status "$SERVICE" --no-pager -n 10 2>/dev/null || true
  local log_path
  log_path="$(resolve_agy_log)"
  if [ -f "$log_path" ]; then
    echo -e "\n=== Recent Daemon Log ($log_path) ==="
    tail -n 15 "$log_path"
  fi
}

do_update() {
  log "Checking/updating Antigravity CLI..."
  local before after
  before=$(agy --version 2>/dev/null || echo "none")
  if agy update 2>&1 | tee -a "$LOG_FILE"; then
    after=$(agy --version 2>/dev/null || echo "none")
    log "Update check complete: before=$before after=$after"
    [ "$before" != "$after" ] && do_restart "cli_updated"
  else
    log "WARN: 'agy update' returned non-zero (may already be latest or require manual installer)."
  fi
}

do_ui_sync() {
  log "Syncing Antigravity UI customizations..."
  "$PYTHON_BIN" "$ROOT/orchestratorOS/agy/ui_mirror.py" 2>&1 | tee -a "$LOG_FILE"
}

do_clean_logs() {
  log "Truncating large Antigravity CLI logs (>20MB)..."
  find "$AGY_DIR" "$AGY_CLI_DIR" -maxdepth 3 -name "*.log" -size +20M -exec truncate -s 5M {} + 2>/dev/null || true
  [ -f "$LOG_FILE" ] && [ "$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)" -gt 10485760 ] && tail -n 2000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
  log "Log cleanup complete."
}

do_clean_sessions() {
  local days="${1:-14}"
  log "Cleaning Antigravity scratch/conversations older than $days days..."
  [ -d "$AGY_DIR/scratch" ] && find "$AGY_DIR/scratch" -mindepth 1 -maxdepth 2 -mtime "+$days" -exec rm -rf {} + 2>/dev/null || true
  log "Session cleanup complete."
}

case "${1:-watchdog}" in
  watchdog)       do_watchdog ;;
  restart)        do_restart "${2:-manual}" ;;
  status)         do_status ;;
  update|cron-update) do_update ;;
  ui-sync)        do_ui_sync ;;
  clean-logs)     do_clean_logs ;;
  clean-sessions) do_clean_sessions "${2:-14}" ;;
  help|--help|-h) echo "Usage: $0 {watchdog|restart|status|update|ui-sync|clean-logs|clean-sessions [days]}" ;;
  *)              echo "Unknown command: $1" >&2; exit 2 ;;
esac
