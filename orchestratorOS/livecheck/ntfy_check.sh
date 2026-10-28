#!/usr/bin/env bash
# ntfy_check.sh — daily health-check ping to verify ntfy delivery channel is alive.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/os_notify.sh
source "$SCRIPT_DIR/../lib/os_notify.sh"

TOPIC="$(ntfy_error_topic)"
HOST="$(hostname)"
UPTIME_STR="$(uptime -p 2>/dev/null || uptime)"
ORCH_STATE="$(systemctl is-active python_orchestrator.service 2>/dev/null || echo "n/a")"
PIA_STATE="$(systemctl is-active pia.service 2>/dev/null || echo "n/a")"
DISK_FREE="$(df -h "$ROOT" | awk 'NR==2 {print $4 " free (" $5 " used)"}')"

TITLE="Daily Heartbeat OK ($HOST)"
BODY="Host: $HOST | Orchestrator: $ORCH_STATE | PIA VPN: $PIA_STATE | Disk: $DISK_FREE | $UPTIME_STR"

ntfy_direct_curl "$TOPIC" default white_check_mark "$TITLE" "$BODY" 10
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ntfy_check sent to topic=$TOPIC"
