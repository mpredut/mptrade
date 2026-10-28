#!/usr/bin/env bash
# deadman_switch.sh — external heartbeat monitor that alerts NTFY_TOPIC_ERROR when the fleet or notify server is dead.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$SCRIPT_DIR/../lib/os_notify.sh"

TOPIC="$(ntfy_error_topic)"
PORT="$(env_get NOTIFY_PORT "9876")"
HOST="$(hostname)"

[ -f "$ROOT/FLEET_PAUSED" ] && exit 0

ISSUES=()
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files python_orchestrator.service >/dev/null 2>&1; then
  if systemctl is-enabled --quiet python_orchestrator.service 2>/dev/null; then
    orch_state="$(systemctl is-active python_orchestrator.service 2>/dev/null || echo "inactive")"
    [ "$orch_state" != "active" ] && [ "$orch_state" != "activating" ] && ISSUES+=("python_orchestrator.service=$orch_state")
  fi
fi

if ! pgrep -f "notify_engine_orchestrator\.py" >/dev/null 2>&1; then
  ISSUES+=("notify_engine_orchestrator.py=DOWN")
elif ! timeout 3 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
  ISSUES+=("notify_engine_orchestrator_port_${PORT}=UNREACHABLE")
fi

STATE_FILE="/tmp/mptrade_deadman_last_alert"
if [ "${#ISSUES[@]}" -eq 0 ]; then
  rm -f "$STATE_FILE" 2>/dev/null || true
  exit 0
fi

NOW="$(date +%s)"
if [ -f "$STATE_FILE" ]; then
  LAST="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"
  [ $((NOW - LAST)) -lt 900 ] && exit 0
fi

MSG="Deadman Switch Alert on ${HOST}: ${ISSUES[*]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $MSG" >&2
if ntfy_direct_curl "$TOPIC" urgent rotating_light "DEADMAN ALERT ($HOST)" "$MSG" 8; then
  echo "$NOW" > "$STATE_FILE"
fi
